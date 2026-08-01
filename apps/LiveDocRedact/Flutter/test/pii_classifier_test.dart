import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:livedocredact/models/read_field.dart';
import 'package:livedocredact/services/pii_classifier.dart';

PiiInputField field(String text, Rect bbox) =>
    PiiInputField(bbox: bbox, text: text, confidence: 0.9);

void main() {
  final classifier = PiiClassifier();

  group('text-only rules', () {
    test('date formats classify as DOB', () {
      expect(classifier.classifyText('12/05/1988'), PiiClass.dob);
      expect(classifier.classifyText('12-05-1988'), PiiClass.dob);
      expect(classifier.classifyText('1988-05-12'), PiiClass.dob);
      expect(classifier.classifyText('12 MAY 1988'), PiiClass.dob);
      expect(classifier.classifyText('MAY 12, 1988'), PiiClass.dob);
      expect(classifier.classifyText('12.05.88'), PiiClass.dob);
    });

    test('ID-number shapes classify as ID', () {
      expect(classifier.classifyText('123-45-6789'), PiiClass.idNumber);
      expect(classifier.classifyText('P1234567'), PiiClass.idNumber);
      expect(classifier.classifyText('123456789'), PiiClass.idNumber);
      expect(classifier.classifyText('DL-9922-AX41'), PiiClass.idNumber);
    });

    test('MRZ lines classify as MRZ', () {
      expect(
        classifier.classifyText(
            'P<UTODOE<<JOHN<MICHAEL<<<<<<<<<<<<<<<<<<<<<<'),
        PiiClass.mrz,
      );
      expect(classifier.classifyText('L898902C36UTO7408122F1204159<<<<<<<8'),
          PiiClass.mrz);
    });

    test('benign document text stays OTHER', () {
      expect(classifier.classifyText('REPUBLIC OF UTOPIA'), PiiClass.other);
      expect(classifier.classifyText('Hello world'), PiiClass.other);
      expect(classifier.classifyText('12345'), PiiClass.other,
          reason: 'short digit runs are not ID numbers');
      expect(classifier.classifyText('POLICY < TERMS'), PiiClass.other,
          reason: 'one stray < in mixed text is not an MRZ');
      expect(classifier.classifyText(''), PiiClass.other);
    });
  });

  group('keyword anchors', () {
    test('inline value: "Name: JOHN DOE" is a NAME field', () {
      final out = classifier.classify(
          [field('Name: JOHN DOE', const Rect.fromLTWH(0, 0, 200, 30))]);
      expect(out.single.piiClass, PiiClass.name);
    });

    test('inline DOB / ID anchors', () {
      final out = classifier.classify([
        field('DOB 12/05/1988', const Rect.fromLTWH(0, 0, 200, 30)),
        field('Passport No: X4711', const Rect.fromLTWH(0, 50, 200, 30)),
      ]);
      expect(out[0].piiClass, PiiClass.dob);
      expect(out[1].piiClass, PiiClass.idNumber);
    });

    test('label-only anchor marks the field to its RIGHT (same line)', () {
      final out = classifier.classify([
        field('Surname', const Rect.fromLTWH(10, 10, 80, 24)),
        field('DOE', const Rect.fromLTWH(100, 10, 80, 24)),
      ]);
      expect(out[0].piiClass, PiiClass.other,
          reason: 'the label itself is not PII — only the value');
      expect(out[1].piiClass, PiiClass.name);
    });

    test('label-only anchor falls back to the field BELOW', () {
      final out = classifier.classify([
        field('ID Number', const Rect.fromLTWH(10, 10, 100, 24)),
        field('XA-99-ZZ', const Rect.fromLTWH(10, 40, 100, 24)),
      ]);
      expect(out[1].piiClass, PiiClass.idNumber,
          reason: 'anchor adjacency must catch values with no ID-like shape');
    });

    test('an anchor does not steal a field that already has a class', () {
      final out = classifier.classify([
        field('Name', const Rect.fromLTWH(10, 10, 60, 24)),
        field('12/05/1988', const Rect.fromLTWH(80, 10, 100, 24)),
        field('JOHN DOE', const Rect.fromLTWH(10, 44, 100, 24)),
      ]);
      expect(out[1].piiClass, PiiClass.dob,
          reason: 'the date keeps its regex class');
      expect(out[2].piiClass, PiiClass.name,
          reason: 'the anchor skips the already-classified date on its right '
              'and pairs with the unclassified field below');
    });

    test('far-away fields are not anchor-paired', () {
      final out = classifier.classify([
        field('Name', const Rect.fromLTWH(10, 10, 60, 24)),
        field('UTOPIA CITY', const Rect.fromLTWH(600, 700, 120, 24)),
      ]);
      expect(out[1].piiClass, PiiClass.other);
    });
  });
}
