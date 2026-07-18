import 'dart:typed_data';

/// One catalog entry in the bundled demo gallery: a precomputed 512-d unit
/// vector (embedded from a real product photo through the REAL exported
/// visualsearch_embed.onnx) plus display metadata and a thumbnail asset path.
class GalleryItem {
  const GalleryItem({
    required this.id,
    required this.label,
    required this.category,
    required this.instance,
    required this.view,
    required this.vector,
    required this.thumbAsset,
  });

  final String id;
  final String label;
  final String category;

  /// Product-instance key (same physical product, different photo). Used for
  /// the "SAME PRODUCT" badge when a match shares the query's instance.
  final String instance;
  final String view;

  /// Unit-norm 512-d embedding. Cosine similarity == a plain dot product.
  final Float32List vector;

  final String thumbAsset;
}

/// A ranked match: a gallery item and its cosine score against the query.
class SearchResult {
  const SearchResult({required this.item, required this.score});
  final GalleryItem item;
  final double score;
}
