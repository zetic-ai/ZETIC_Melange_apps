package ai.zetic.demo.imageto3d

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.opengl.Matrix
import android.view.MotionEvent
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sin
import kotlin.math.tan

/**
 * Interactive relief viewer: photo-textured mesh or colored point cloud,
 * turntable idle sway (stops on touch), one-finger orbit, two-finger pinch
 * zoom. Mirrors the iOS SceneKit view.
 */
class Relief3DView(context: Context) : GLSurfaceView(context) {
    private val renderer = ReliefRenderer()
    private var lastX = 0f
    private var lastY = 0f
    private var lastSpan = 0f

    init {
        setEGLContextClientVersion(3)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun setMesh(mesh: MeshData, texture: Bitmap) {
        queueEvent { renderer.setMesh(mesh, texture) }
    }

    fun setMode(points: Boolean) {
        queueEvent { renderer.showPoints = points }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                renderer.userControlled = true
                lastX = event.x
                lastY = event.y
            }
            MotionEvent.ACTION_POINTER_DOWN -> lastSpan = span(event)
            MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount >= 2) {
                    val s = span(event)
                    if (lastSpan > 0) renderer.zoom *= s / lastSpan
                    renderer.zoom = renderer.zoom.coerceIn(0.4f, 4f)
                    lastSpan = s
                } else {
                    renderer.yaw += (event.x - lastX) * 0.4f
                    renderer.pitch = (renderer.pitch + (event.y - lastY) * 0.3f)
                        .coerceIn(-80f, 80f)
                    lastX = event.x
                    lastY = event.y
                }
            }
        }
        return true
    }

    private fun span(e: MotionEvent): Float {
        val dx = e.getX(0) - e.getX(1)
        val dy = e.getY(0) - e.getY(1)
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
}

private class ReliefRenderer : GLSurfaceView.Renderer {
    @Volatile var showPoints = false
    @Volatile var userControlled = false
    @Volatile var yaw = 0f
    @Volatile var pitch = 0f
    @Volatile var zoom = 1f

    private var pendingMesh: MeshData? = null
    private var pendingTexture: Bitmap? = null

    private var meshProgram = 0
    private var pointProgram = 0
    private var backgroundProgram = 0
    private var vbo = IntArray(5)   // positions, uvs, colors, triIndices, pointIndices
    private var textureID = 0
    private var triangleCount = 0
    private var pointCount = 0
    private var center = floatArrayOf(0f, 0f, 0f)
    private var radius = 1f
    private var aspect = 1f
    private var startNanos = System.nanoTime()

    private val mvp = FloatArray(16)
    private val proj = FloatArray(16)
    private val view = FloatArray(16)
    private val model = FloatArray(16)
    private val tmp = FloatArray(16)

    fun setMesh(mesh: MeshData, texture: Bitmap) {
        pendingMesh = mesh
        pendingTexture = texture
        userControlled = false
        yaw = 0f
        pitch = 0f
        zoom = 1f
        startNanos = System.nanoTime()
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        meshProgram = buildProgram(MESH_VS, MESH_FS)
        pointProgram = buildProgram(POINT_VS, POINT_FS)
        backgroundProgram = buildProgram(BG_VS, BG_FS)
        GLES30.glGenBuffers(vbo.size, vbo, 0)
        val tex = IntArray(1)
        GLES30.glGenTextures(1, tex, 0)
        textureID = tex[0]
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        aspect = width.toFloat() / height
    }

    override fun onDrawFrame(gl: GL10?) {
        uploadPendingMesh()
        GLES30.glClearColor(0.05f, 0.05f, 0.05f, 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        drawBackground()
        if (triangleCount == 0 && pointCount == 0) return

        // Camera: fit bounding sphere, slight elevation (mirrors iOS).
        val fov = AppConfig.FOV_Y_DEGREES
        val distance = radius / tan(Math.toRadians(fov / 2.0)).toFloat() * 1.1f / zoom
        val elevation = 0.18f
        Matrix.perspectiveM(proj, 0, fov, aspect, 0.01f, (distance + radius) * 4)
        Matrix.setLookAtM(view, 0,
            0f, distance * sin(elevation), distance * kotlin.math.cos(elevation),
            0f, 0f, 0f, 0f, 1f, 0f)

        // Model: pivot at mesh center; idle sway until the user takes over.
        val sway = if (userControlled) 0f else {
            val t = (System.nanoTime() - startNanos) / 1e9f
            35f * sin(t * (2f * Math.PI.toFloat() / 8f))
        }
        Matrix.setIdentityM(model, 0)
        Matrix.rotateM(model, 0, pitch, 1f, 0f, 0f)
        Matrix.rotateM(model, 0, yaw + sway, 0f, 1f, 0f)
        Matrix.translateM(model, 0, -center[0], -center[1], -center[2])

        Matrix.multiplyMM(tmp, 0, view, 0, model, 0)
        Matrix.multiplyMM(mvp, 0, proj, 0, tmp, 0)

        if (!showPoints && triangleCount > 0) drawMesh() else drawPoints()
    }

    private fun drawMesh() {
        GLES30.glUseProgram(meshProgram)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(meshProgram, "uMVP"), 1, false, mvp, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureID)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(meshProgram, "uTexture"), 0)

        bindAttrib(meshProgram, "aPos", vbo[0], 3)
        bindAttrib(meshProgram, "aUV", vbo[1], 2)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo[3])
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, triangleCount, GLES30.GL_UNSIGNED_INT, 0)
    }

    private fun drawPoints() {
        GLES30.glUseProgram(pointProgram)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(pointProgram, "uMVP"), 1, false, mvp, 0)
        bindAttrib(pointProgram, "aPos", vbo[0], 3)
        bindAttrib(pointProgram, "aColor", vbo[2], 3)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, vbo[4])
        GLES30.glDrawElements(GLES30.GL_POINTS, pointCount, GLES30.GL_UNSIGNED_INT, 0)
    }

    private fun drawBackground() {
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glUseProgram(backgroundProgram)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
    }

    private fun bindAttrib(program: Int, name: String, buffer: Int, size: Int) {
        val loc = GLES30.glGetAttribLocation(program, name)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, buffer)
        GLES30.glEnableVertexAttribArray(loc)
        GLES30.glVertexAttribPointer(loc, size, GLES30.GL_FLOAT, false, 0, 0)
    }

    private fun uploadPendingMesh() {
        val mesh = pendingMesh ?: return
        val texture = pendingTexture ?: return
        pendingMesh = null
        pendingTexture = null

        uploadFloats(vbo[0], mesh.positions)
        uploadFloats(vbo[1], mesh.uvs)
        uploadFloats(vbo[2], mesh.pointColors)
        uploadInts(vbo[3], mesh.triangleIndices, element = true)
        uploadInts(vbo[4], mesh.pointIndices, element = true)
        triangleCount = mesh.triangleIndices.size
        pointCount = mesh.pointIndices.size
        center = mesh.center
        radius = mesh.radius

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureID)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, texture, 0)
        GLES30.glGenerateMipmap(GLES30.GL_TEXTURE_2D)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER,
                               GLES30.GL_LINEAR_MIPMAP_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER,
                               GLES30.GL_LINEAR)
    }

    private fun uploadFloats(buffer: Int, data: FloatArray) {
        val bb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        bb.asFloatBuffer().put(data)
        bb.rewind()
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, buffer)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, data.size * 4, bb, GLES30.GL_STATIC_DRAW)
    }

    private fun uploadInts(buffer: Int, data: IntArray, element: Boolean) {
        val target = if (element) GLES30.GL_ELEMENT_ARRAY_BUFFER else GLES30.GL_ARRAY_BUFFER
        val bb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        bb.asIntBuffer().put(data)
        bb.rewind()
        GLES30.glBindBuffer(target, buffer)
        GLES30.glBufferData(target, data.size * 4, bb, GLES30.GL_STATIC_DRAW)
    }

    private fun buildProgram(vs: String, fs: String): Int {
        fun compile(type: Int, src: String): Int {
            val shader = GLES30.glCreateShader(type)
            GLES30.glShaderSource(shader, src)
            GLES30.glCompileShader(shader)
            val ok = IntArray(1)
            GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, ok, 0)
            check(ok[0] != 0) { "shader compile failed: ${GLES30.glGetShaderInfoLog(shader)}" }
            return shader
        }
        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, compile(GLES30.GL_VERTEX_SHADER, vs))
        GLES30.glAttachShader(program, compile(GLES30.GL_FRAGMENT_SHADER, fs))
        GLES30.glLinkProgram(program)
        val ok = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, ok, 0)
        check(ok[0] != 0) { "program link failed: ${GLES30.glGetProgramInfoLog(program)}" }
        return program
    }

    companion object {
        private const val MESH_VS = """#version 300 es
            uniform mat4 uMVP;
            in vec3 aPos;
            in vec2 aUV;
            out vec2 vUV;
            void main() { vUV = aUV; gl_Position = uMVP * vec4(aPos, 1.0); }"""

        private const val MESH_FS = """#version 300 es
            precision mediump float;
            uniform sampler2D uTexture;
            in vec2 vUV;
            out vec4 frag;
            void main() { frag = vec4(texture(uTexture, vUV).rgb, 1.0); }"""

        private const val POINT_VS = """#version 300 es
            uniform mat4 uMVP;
            in vec3 aPos;
            in vec3 aColor;
            out vec3 vColor;
            void main() {
                vColor = aColor;
                gl_Position = uMVP * vec4(aPos, 1.0);
                gl_PointSize = clamp(14.0 / gl_Position.w, 3.0, 18.0);
            }"""

        private const val POINT_FS = """#version 300 es
            precision mediump float;
            in vec3 vColor;
            out vec4 frag;
            void main() { frag = vec4(vColor, 1.0); }"""

        // Fullscreen vertical gradient so occlusion holes read as shadow.
        private const val BG_VS = """#version 300 es
            out vec2 vPos;
            void main() {
                vec2 p = vec2(float(gl_VertexID % 2) * 2.0 - 1.0,
                              float(gl_VertexID / 2) * 2.0 - 1.0);
                vPos = p;
                gl_Position = vec4(p, 0.9999, 1.0);
            }"""

        private const val BG_FS = """#version 300 es
            precision mediump float;
            in vec2 vPos;
            out vec4 frag;
            void main() {
                float t = vPos.y * 0.5 + 0.5;
                frag = vec4(vec3(mix(0.05, 0.16, t)), 1.0);
            }"""
    }
}
