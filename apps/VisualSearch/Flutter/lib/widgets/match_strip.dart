import 'package:flutter/material.dart';

import '../models/gallery_item.dart';
import '../theme.dart';

/// Horizontal strip of the top-K nearest catalog matches: thumbnail, cosine
/// score, label, and a SAME-PRODUCT badge when a match shares the query's
/// (best match's) product instance.
class MatchStrip extends StatelessWidget {
  const MatchStrip({super.key, required this.results});

  final List<SearchResult> results;

  @override
  Widget build(BuildContext context) {
    if (results.isEmpty) {
      return const SizedBox.shrink();
    }
    final String topInstance = results.first.item.instance;
    return SizedBox(
      height: 168,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        itemCount: results.length,
        separatorBuilder: (_, _) => const SizedBox(width: 10),
        itemBuilder: (context, i) {
          final r = results[i];
          final bool same = i != 0 && r.item.instance == topInstance;
          return _MatchCard(result: r, rank: i + 1, sameProduct: same);
        },
      ),
    );
  }
}

class _MatchCard extends StatelessWidget {
  const _MatchCard(
      {required this.result, required this.rank, required this.sameProduct});

  final SearchResult result;
  final int rank;
  final bool sameProduct;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 118,
      decoration: BoxDecoration(
        color: SnapColors.surface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
            color: rank == 1 ? SnapColors.accent : Colors.white12,
            width: rank == 1 ? 2 : 1),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.asset(result.item.thumbAsset, fit: BoxFit.cover),
                Positioned(
                  left: 4,
                  top: 4,
                  child: _chip('cos ${result.score.toStringAsFixed(3)}',
                      SnapColors.accent, Colors.black),
                ),
                if (sameProduct)
                  Positioned(
                    right: 4,
                    top: 4,
                    child: _chip('SAME', SnapColors.amber, Colors.black),
                  ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 5),
            child: Text(
              result.item.label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: SnapColors.textLo, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }

  Widget _chip(String text, Color bg, Color fg) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
        decoration:
            BoxDecoration(color: bg, borderRadius: BorderRadius.circular(4)),
        child: Text(text,
            style: TextStyle(
                color: fg, fontSize: 10, fontWeight: FontWeight.w700)),
      );
}
