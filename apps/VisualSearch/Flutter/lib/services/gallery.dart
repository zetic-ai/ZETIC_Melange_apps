import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import '../models/gallery_item.dart';

/// The bundled demo catalog: precomputed 512-d unit vectors (from the REAL
/// exported visualsearch_embed.onnx over 60 real product photos) + thumbnails.
/// On-device matching is a plain dot product (both sides are unit-norm), so the
/// cosine scores shown are meaningful.
class Gallery {
  Gallery._(this.items, this.dim);

  final List<GalleryItem> items;
  final int dim;

  static const String _jsonAsset = 'assets/gallery/gallery.json';
  static const String _thumbDir = 'assets/gallery/thumbs';

  bool get isEmpty => items.isEmpty;

  /// Load + parse the gallery JSON from the app bundle.
  static Future<Gallery> load() async {
    final raw = await rootBundle.loadString(_jsonAsset);
    final Map<String, dynamic> data = json.decode(raw) as Map<String, dynamic>;
    final int dim = (data['dim'] as num).toInt();
    final List<dynamic> rawItems = data['items'] as List<dynamic>;

    final items = <GalleryItem>[];
    for (final dynamic e in rawItems) {
      final m = e as Map<String, dynamic>;
      final List<dynamic> v = m['vector'] as List<dynamic>;
      final vec = Float32List(v.length);
      for (var i = 0; i < v.length; i++) {
        vec[i] = (v[i] as num).toDouble();
      }
      final id = m['id'] as String;
      items.add(GalleryItem(
        id: id,
        label: m['label'] as String,
        category: m['category'] as String,
        instance: m['instance'] as String,
        view: m['view'] as String,
        vector: vec,
        thumbAsset: '$_thumbDir/$id.jpg',
      ));
    }
    return Gallery._(items, dim);
  }

  /// Rank the gallery against a unit-norm query embedding by cosine == dot
  /// product; return the top-[k] descending. Pure so it can be unit-tested and,
  /// if needed, run in an isolate.
  List<SearchResult> topK(Float32List query, {int k = 5}) {
    return rankByDot(items, query, k: k);
  }
}

/// Cosine (dot-product) ranking over unit vectors. Extracted + top-level for
/// direct testing.
List<SearchResult> rankByDot(
  List<GalleryItem> items,
  Float32List query, {
  int k = 5,
}) {
  final scored = <SearchResult>[];
  for (final it in items) {
    final v = it.vector;
    final int n = v.length < query.length ? v.length : query.length;
    double dot = 0;
    for (var i = 0; i < n; i++) {
      dot += v[i] * query[i];
    }
    scored.add(SearchResult(item: it, score: dot));
  }
  scored.sort((a, b) => b.score.compareTo(a.score));
  return scored.length <= k ? scored : scored.sublist(0, k);
}
