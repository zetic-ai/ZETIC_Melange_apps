import 'package:flutter_test/flutter_test.dart';
import 'package:sensorforecastts/services/data_feed.dart';

void main() {
  group('SensorFeed determinism', () {
    test('same seed + mode reproduces the identical series', () {
      final a = SensorFeed(mode: FeedMode.industrial, seed: 7);
      final b = SensorFeed(mode: FeedMode.industrial, seed: 7);
      for (var i = 0; i < 2000; i++) {
        expect(a.next(), b.next(), reason: 'sample $i diverged');
      }
    });

    test('different seeds diverge', () {
      final a = SensorFeed(mode: FeedMode.lab, seed: 1);
      final b = SensorFeed(mode: FeedMode.lab, seed: 2);
      var identical = true;
      for (var i = 0; i < 50; i++) {
        if (a.next() != b.next()) identical = false;
      }
      expect(identical, isFalse);
    });

    test('industrial signal stays in a plausible plant-sensor range', () {
      final f = SensorFeed(mode: FeedMode.industrial, seed: 7);
      for (var i = 0; i < SensorFeed.loopLength; i++) {
        final v = f.next();
        expect(v, inInclusiveRange(55.0, 125.0),
            reason: 'sample $i out of range: $v');
      }
    });
  });

  group('SensorFeed injections', () {
    test('spike corrupts exactly one sample by +25', () {
      final clean = SensorFeed(mode: FeedMode.lab, seed: 9);
      final dirty = SensorFeed(mode: FeedMode.lab, seed: 9);
      for (var i = 0; i < 100; i++) {
        clean.next();
        dirty.next();
      }
      dirty.inject(InjectionKind.spike);
      expect(dirty.next(), closeTo(clean.next() + 25.0, 1e-9),
          reason: 'spiked sample');
      expect(dirty.next(), clean.next(), reason: 'next sample is clean again');
    });

    test('level shift holds +12 for exactly 60 samples', () {
      final clean = SensorFeed(mode: FeedMode.lab, seed: 3);
      final dirty = SensorFeed(mode: FeedMode.lab, seed: 3);
      dirty.inject(InjectionKind.levelShift);
      for (var i = 0; i < 60; i++) {
        expect(dirty.next(), closeTo(clean.next() + 12.0, 1e-9),
            reason: 'shifted sample $i');
      }
      expect(dirty.next(), clean.next(), reason: 'shift released after 60');
    });

    test('noise burst perturbs samples then releases after 40', () {
      // Noise burst draws extra RNG numbers, so clean/dirty streams diverge
      // during the burst; assert perturbation happens and the burst expires.
      final f = SensorFeed(mode: FeedMode.lab, seed: 5);
      f.inject(InjectionKind.noiseBurst);
      var perturbed = 0;
      for (var i = 0; i < 40; i++) {
        final v = f.next();
        // Lab signal without burst lives within ~50 +/- 13; count clear
        // outliers as evidence the burst is active.
        if ((v - 50.0).abs() > 13.0) perturbed++;
      }
      expect(perturbed, greaterThan(0),
          reason: 'burst must visibly perturb the signal');
      // After expiry the feed keeps producing plausible lab values.
      for (var i = 0; i < 100; i++) {
        expect(f.next(), inInclusiveRange(20.0, 80.0));
      }
    });
  });
}
