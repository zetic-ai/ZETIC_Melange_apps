import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/models/gallery_item.dart';
import 'package:snapseek/services/gallery.dart';

Float32List unit(List<double> v) {
  final f = Float32List.fromList(v);
  double n = 0;
  for (final x in f) {
    n += x * x;
  }
  n = math.sqrt(n);
  for (var i = 0; i < f.length; i++) {
    f[i] = f[i] / n;
  }
  return f;
}

GalleryItem item(String id, Float32List v, {String instance = 'i'}) =>
    GalleryItem(
      id: id,
      label: id,
      category: 'c',
      instance: instance,
      view: 'v',
      vector: v,
      thumbAsset: 'x',
    );

void main() {
  group('gallery cosine (dot-product) ranking', () {
    final a = unit([1, 0, 0, 0]);
    final b = unit([0.9, 0.1, 0, 0]);
    final c = unit([0, 1, 0, 0]);
    final d = unit([0, 0, 1, 0]);
    final items = [item('a', a), item('b', b), item('c', c), item('d', d)];

    test('identical unit vector scores exactly 1.0 (cosine(v,v)=1)', () {
      final res = rankByDot(items, a, k: 4);
      expect(res.first.item.id, 'a');
      expect(res.first.score, closeTo(1.0, 1e-6));
    });

    test('results are sorted by descending cosine', () {
      final res = rankByDot(items, a, k: 4);
      for (var i = 1; i < res.length; i++) {
        expect(res[i - 1].score, greaterThanOrEqualTo(res[i].score));
      }
      // a (1.0) then b (~0.994) then the orthogonals (~0).
      expect(res[0].item.id, 'a');
      expect(res[1].item.id, 'b');
    });

    test('top-K truncates to k', () {
      expect(rankByDot(items, a, k: 2), hasLength(2));
      expect(rankByDot(items, a, k: 10), hasLength(4));
    });

    test('orthogonal vectors score ~0', () {
      final res = rankByDot([item('c', c)], a, k: 1);
      expect(res.single.score, closeTo(0.0, 1e-6));
    });
  });
}
