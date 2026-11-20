"""Registration evidence regressions; no network, detector or training."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from src.models.registry import Registry, canonical, prepare, raw_dvc_identity


def digest(data):
    return hashlib.sha256(data).hexdigest()


class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'best.pt'
        self.checkpoint.write_bytes(b'fixture-checkpoint')
        self.manifest = self.root / 'manifest.csv'
        self.manifest.write_text('image,date,split\n'
            'data/raw/train/images/train.jpg,2012-01-01,train\n'
            'data/raw/valid/images/one.jpg,2012-01-02,valid\n'
            'data/raw/valid/images/two.jpg,2012-01-02,valid\n'
            'data/raw/test/images/test.jpg,2012-01-03,test\n', encoding='utf-8')
        self.spec = {'split': 'valid', 'images': 2,
                     'manifest_sha256': digest(self.manifest.read_bytes()),
                     'confidence': .25, 'nms_iou': .7, 'matching_iou': .5, 'imgsz': 640}
        self.protocol = self.root / 'protocol.yaml'
        self.protocol.write_text(json.dumps(self.spec), encoding='utf-8')
        self.csv = self.root / 'count.csv'
        self.summary = self.root / 'summary.json'
        self.env = self.root / 'env.json'
        self.archive = self.root / 'export.zip'
        self.raw = b'outs:\n- md5: ' + b'a' * 32 + b'.dir\n  path: raw\n  size: 20\n  nfiles: 4\n'
        self.names = ['one.jpg', 'two.jpg']
        self.code = b'fixture-evaluator'
        self.write_archive()
        self.write_counts()

    def write_archive(self, raw=None):
        raw = self.raw if raw is None else raw
        artifacts = {'training/weights/best.pt': self.checkpoint.read_bytes(),
                     'provenance/environment.json': b'{"ultralytics":"8.4.201"}',
                     'provenance/train_b.yaml': b'model: yolov8s.pt\n',
                     'provenance/raw.dvc': raw}
        metadata = {'format': 'parkvision-run-v1', 'status': 'FINISHED',
                    'source_run_id': 'a' * 32,
                    'tags': {'type': 'training', 'test_evaluated': 'false',
                             'git_commit': 'b' * 40,
                             'manifest_sha256': digest(self.manifest.read_bytes())},
                    'artifact_hashes': {key: digest(value) for key, value in artifacts.items()},
                    'metrics': [{'key': 'best_val/metrics/mAP50-95_B_', 'value': .93}]}
        with ZipFile(self.archive, 'w') as z:
            z.writestr('metadata.json', json.dumps(metadata))
            for name, data in artifacts.items():
                z.writestr('artifacts/' + name, data)

    def write_counts(self):
        with self.csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'libres_annotees', 'libres_predites',
                             'occupees_annotees', 'occupees_predites'])
            writer.writerows((name, 1, 1, 1, 1) for name in self.names)
        self.summary.write_text(json.dumps({'images_evaluees': len(self.names),
            'mae_libres': 0, 'mae_occupees': 0, 'poids_sha256': digest(self.checkpoint.read_bytes()),
            'commit_git': 'c' * 40, 'seuil_confiance': .25, 'seuil_nms_iou': .7,
            'seuil_appariement_iou': .5, 'imgsz': 640}), encoding='utf-8')
        self.env.write_text(json.dumps({'git_commit': 'c' * 40,
            'analysis_code_sha256': digest(self.code),
            'validation_image_names_sha256': digest(canonical(sorted(self.names)).encode()),
            'raw_dvc_sha256': digest(self.raw), 'ultralytics_version': '8.4.201'}), encoding='utf-8')

    def prepare(self, with_env=True):
        with patch('src.models.registry.subprocess.check_output', return_value=self.code):
            return prepare(self.archive, self.checkpoint, self.manifest, self.csv,
                           self.summary, 'd' * 32, evaluation_protocol=self.protocol,
                           evaluation_environment=self.env if with_env else None)

    def test_complete_evidence_passes(self):
        record = self.prepare()
        self.assertTrue(record['validation_contract']['complete_validation_set'])
        self.assertTrue(record['comparison_verified'])

    def test_missing_evaluation_environment_remains_unverified(self):
        self.assertFalse(self.prepare(with_env=False)['comparison_verified'])

    def test_partial_consistent_counts_rejected(self):
        self.names = ['one.jpg']
        self.write_counts()
        with self.assertRaisesRegex(ValueError, 'complete validation'):
            self.prepare()

    def test_test_or_unknown_image_cannot_replace_valid_image(self):
        for image in ('test.jpg', 'other.jpg'):
            with self.subTest(image=image):
                self.names = ['one.jpg', image]
                self.write_counts()
                with self.assertRaisesRegex(ValueError, 'complete validation'):
                    self.prepare()

    def test_duplicate_count_image_rejected(self):
        self.names = ['one.jpg', 'one.jpg']
        self.write_counts()
        with self.assertRaisesRegex(ValueError, 'duplicate image'):
            self.prepare()

    def test_changed_protocol_rejected(self):
        original = json.loads(self.summary.read_text())
        for key, value in [('seuil_confiance', .9), ('seuil_nms_iou', .5),
                           ('seuil_appariement_iou', .9), ('imgsz', 320)]:
            with self.subTest(key=key):
                self.summary.write_text(json.dumps({**original, key: value}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
                    self.prepare()

    def test_manifest_must_match_approved_contract(self):
        self.manifest.write_text(self.manifest.read_text().replace('one.jpg', 'other.jpg'), encoding='utf-8')
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'fixed protocol'):
            self.prepare()

    def test_same_data_line_endings_keep_comparison_protocol(self):
        before = self.prepare()
        self.write_archive(self.raw.replace(b'\n', b'\r\n'))
        after = self.prepare()
        self.assertEqual(before['protocol_id'], after['protocol_id'])
        self.assertNotEqual(before['training_raw_dvc_file_sha256'], after['training_raw_dvc_file_sha256'])

    def test_different_dvc_tree_is_rejected_by_environment(self):
        self.write_archive(self.raw.replace(b'a' * 32, b'b' * 32))
        with self.assertRaisesRegex(ValueError, 'raw_dvc_sha256 differs'):
            self.prepare()

    def test_invalid_dvc_metadata_rejected(self):
        self.write_archive(b'not DVC metadata')
        with self.assertRaisesRegex(ValueError, 'DVC output'):
            self.prepare()

    def test_semantic_identity_ignores_yaml_formatting(self):
        self.assertEqual(raw_dvc_identity(self.raw),
                         raw_dvc_identity(self.raw.replace(b'\n', b'\r\n')))


class PaginationTests(unittest.TestCase):
    def test_versions_reads_all_pages(self):
        class Page(list):
            token = None
        first, second = Page([SimpleNamespace(version='1')]), Page([SimpleNamespace(version='2')])
        first.token = 'next'
        from unittest.mock import Mock
        client = Mock()
        client.search_registered_models.return_value = [SimpleNamespace(name='ParkVision-Detector')]
        client.search_model_versions.side_effect = [first, second]
        result = Registry(client).versions()
        self.assertEqual([v.version for v in result], ['1', '2'])
        self.assertEqual(client.search_model_versions.call_args.kwargs, {'page_token': 'next'})
