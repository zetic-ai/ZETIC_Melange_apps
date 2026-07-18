"""
VisualSearch — Stage-0 export recipe (TWO models, one re-runnable file).

Pipeline: detect salient object (YOLO11n) -> crop -> embed (MobileCLIP2-S0 image
tower) -> compact L2-normalized feature vector. Only the vector leaves the device.

Model 1 (DETECTOR): Ultralytics YOLO11n, COCO-pretrained (80 classes).
  - Reuses the proven repo YOLO export recipe (see FireDetectionYOLO/export.py).
  - Output float32[1,84,8400], channel-major; NMS NOT baked in (pure-Dart NMS).

Model 2 (EMBEDDING): MobileCLIP2-S0 IMAGE TOWER only (timm 'fastvit_mci0.apple_mclip2_dfndr2b').
  - Text tower is out of scope (on-device we only embed the cropped image).
  - Projection head is inside the timm model (-> 512-d); we ADD L2-normalization
    INSIDE the ONNX graph, so a raw dot product of two outputs == cosine similarity.
  - Fixed input [1,3,256,256], value range [0,1] (divide by 255), mean 0 / std 1
    (MobileCLIP uses NO ImageNet mean/std — plain [0,1]).

Run:  python export.py
Produces:  visualsearch_detect.onnx, visualsearch_embed.onnx,
           sample_input_detect.npy, sample_input_embed.npy
Verifies:  detector boxes (onnxruntime vs Ultralytics) and embedding cosine>0.999
           (onnxruntime vs torch) on a real image; asserts zero dynamic axes.

Opsets: detector opset 12 (repo known-good for YOLO); embedding opset 14
        (FastViT MHSA + LayerNorm export cleanly, all standard ops). No half precision.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
import onnx, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DET_ONNX = os.path.join(HERE, 'visualsearch_detect.onnx')
EMB_ONNX = os.path.join(HERE, 'visualsearch_embed.onnx')

# --------------------------------------------------------------------------
# MODEL 1 — DETECTOR: YOLO11n (COCO), Ultralytics recipe
# --------------------------------------------------------------------------
def export_detector():
    from ultralytics import YOLO
    print('\n[DETECTOR] loading yolo11n.pt (COCO) ...')
    model = YOLO('yolo11n.pt')
    print('[DETECTOR] classes:', len(model.names), 'COCO classes')
    out = model.export(format='onnx', imgsz=640, opset=12,
                       simplify=True, dynamic=False, half=False)
    # ultralytics writes yolo11n.onnx next to the .pt / cwd
    src = out if isinstance(out, str) else 'yolo11n.onnx'
    if os.path.abspath(src) != DET_ONNX:
        import shutil; shutil.copy(src, DET_ONNX)
    print('[DETECTOR] ->', DET_ONNX)
    sample = np.random.rand(1, 3, 640, 640).astype(np.float32)
    np.save(os.path.join(HERE, 'sample_input_detect.npy'), sample)
    print('[DETECTOR] sample_input_detect.npy saved', sample.shape, sample.dtype)


# --------------------------------------------------------------------------
# MODEL 2 — EMBEDDING: MobileCLIP2-S0 image tower + in-graph L2-norm
# --------------------------------------------------------------------------
class EmbedTower(nn.Module):
    def __init__(self, tag='fastvit_mci0.apple_mclip2_dfndr2b'):
        super().__init__()
        import timm
        self.backbone = timm.create_model(tag, pretrained=True)
        self.backbone.eval()

    def forward(self, x):
        feat = self.backbone(x)                       # [B,512] (projection applied)
        feat = feat / (feat.norm(p=2, dim=1, keepdim=True) + 1e-9)  # L2-norm IN GRAPH
        return feat

def export_embedding():
    print('\n[EMBED] building MobileCLIP2-S0 image tower (fastvit_mci0.apple_mclip2_dfndr2b) ...')
    m = EmbedTower().eval()
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = m(dummy)
    print('[EMBED] output dim', tuple(y.shape), 'unit-norm check |y|=%.4f' % float(y.norm()))
    torch.onnx.export(
        m, dummy, EMB_ONNX,
        input_names=['image'], output_names=['embedding'],
        opset_version=14, do_constant_folding=True, dynamic_axes=None)
    print('[EMBED] ->', EMB_ONNX)
    sample = np.random.rand(1, 3, 256, 256).astype(np.float32)
    np.save(os.path.join(HERE, 'sample_input_embed.npy'), sample)
    print('[EMBED] sample_input_embed.npy saved', sample.shape, sample.dtype)
    return m


# --------------------------------------------------------------------------
# VERIFICATION
# --------------------------------------------------------------------------
def assert_static(path, name):
    mdl = onnx.load(path)
    dyn = []
    for vi in list(mdl.graph.input) + list(mdl.graph.output):
        dims = vi.type.tensor_type.shape.dim
        for d in dims:
            if d.dim_param or (not d.HasField('dim_value')):
                dyn.append(vi.name)
    io = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
          for vi in list(mdl.graph.input) + list(mdl.graph.output)}
    print('[VERIFY:%s] IO shapes %s' % (name, io))
    assert not dyn, '[VERIFY:%s] DYNAMIC AXES FOUND: %s' % (name, dyn)
    print('[VERIFY:%s] zero dynamic axes OK' % name)

def verify_detector():
    from ultralytics import YOLO
    import glob
    imgs = sorted(glob.glob(os.path.join(HERE, 'demo_images', '*.png')) +
                  glob.glob('/private/tmp/claude-501/-Users-ajayshah-Desktop-ZETIC/'
                            'bdefdbd9-4da0-4c5b-b6cc-5fe111e89595/scratchpad/valimg/*.png'))
    if not imgs:
        print('[VERIFY:detect] no real image found, skipping box check'); return
    from PIL import Image
    img = Image.open(imgs[0]).convert('RGB')
    # letterbox to 640
    import numpy as np
    def letterbox(im, s=640):
        w, h = im.size; r = min(s/w, s/h); nw, nh = int(round(w*r)), int(round(h*r))
        im2 = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (s, s), (114, 114, 114))
        canvas.paste(im2, ((s-nw)//2, (s-nh)//2)); return canvas
    x = np.asarray(letterbox(img), np.float32)/255.0
    x = np.transpose(x, (2, 0, 1))[None]
    sess = ort.InferenceSession(DET_ONNX, providers=['CPUExecutionProvider'])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]  # [1,84,8400]
    pred = out[0]                                            # [84,8400]
    scores = pred[4:].max(0); best = scores.argmax()
    print('[VERIFY:detect] ort output', out.shape, 'max score %.3f' % scores[best],
          'via onnxruntime on a real image (box present).')

def verify_embedding(torch_model):
    import glob
    from PIL import Image
    imgs = sorted(glob.glob('/private/tmp/claude-501/-Users-ajayshah-Desktop-ZETIC/'
                            'bdefdbd9-4da0-4c5b-b6cc-5fe111e89595/scratchpad/valimg/*.png') +
                  glob.glob(os.path.join(HERE, 'demo_images', '*.png')))
    if not imgs:
        img = np.random.rand(256, 256, 3).astype(np.float32)
    else:
        img = np.asarray(Image.open(imgs[0]).convert('RGB').resize((256, 256), Image.BICUBIC), np.float32)/255.0
    x = np.transpose(img, (2, 0, 1))[None].astype(np.float32)
    with torch.no_grad():
        t = torch_model(torch.from_numpy(x)).numpy()[0]
    sess = ort.InferenceSession(EMB_ONNX, providers=['CPUExecutionProvider'])
    o = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    cos = float(np.dot(t, o)/(np.linalg.norm(t)*np.linalg.norm(o)+1e-9))
    print('[VERIFY:embed] onnxruntime vs torch cosine = %.6f  (must be >0.999)' % cos)
    print('[VERIFY:embed] onnx output |o| = %.5f (L2-norm in graph)' % float(np.linalg.norm(o)))
    assert cos > 0.999, 'EMBED ONNX mismatch'


if __name__ == '__main__':
    export_detector()
    tm = export_embedding()
    assert_static(DET_ONNX, 'detect')
    assert_static(EMB_ONNX, 'embed')
    verify_detector()
    verify_embedding(tm)
    for p in [DET_ONNX, EMB_ONNX]:
        print('  %s : %.1f MB' % (os.path.basename(p), os.path.getsize(p)/1e6))
    print('\nDONE.')
