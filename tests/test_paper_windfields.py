"""Scientific invariants for the one-off paper evaluation."""
import math

import pytest
import torch

from scripts.evaluate_paper_windfields import aggregate, scene_statistics


def test_physical_error_and_sar_mask():
    # A normalized residual of .025 with a scale of 80 is a physical 2 m/s error.
    target = torch.tensor([[[10., 20., float('nan'), 0.]]])
    prediction = target + .025 * 80
    prediction[0, 0, 3] = float('nan')
    stats = scene_statistics(prediction, target, torch.tensor([[[True, True, True, False]]]))
    assert stats == dict(valid_pixels=2, absolute_error_sum=4., squared_error_sum=8.)
    result = aggregate([dict(storm_id='A', **stats)])
    assert result['l1_ms'] == 2.
    assert result['psnr_db'] == pytest.approx(20 * math.log10(79.8 / 2))


def test_nonfinite_predictions_cannot_shrink_support():
    with pytest.raises(ValueError, match='Nonfinite prediction'):
        scene_statistics(torch.tensor([float('nan')]), torch.tensor([10.]), torch.tensor([True]))
    with pytest.raises(ValueError, match='No finite'):
        scene_statistics(torch.tensor([0.]), torch.tensor([10.]), torch.tensor([False]))


def test_pooling_weights_pixels_and_is_partition_invariant():
    target = torch.zeros(2, 1, 2, 2)
    pred = torch.tensor([[[[1., 1.], [1., 1.]]], [[[3., 3.], [3., 3.]]]])
    mask = torch.tensor([[[[True, False], [False, False]]], [[[True, True], [True, True]]]])
    scenes = [dict(storm_id='A', **scene_statistics(p, t, m)) for p,t,m in zip(pred,target,mask)]
    combined = dict(storm_id='A', **scene_statistics(pred,target,mask))
    per_scene, whole = aggregate(scenes), aggregate([combined])
    for key in ('l1_ms', 'mse_ms2', 'psnr_db', 'valid_pixels'):
        assert per_scene[key] == pytest.approx(whole[key])
    assert per_scene['l1_ms'] == pytest.approx(13/5)
    assert per_scene['mse_ms2'] == pytest.approx(37/5)


def test_empty_subset_is_unavailable_not_zero():
    result = aggregate([])
    assert result['l1_ms'] is None and result['psnr_db'] is None
    assert result['availability_reason'] == 'empty_subset'


def test_audit_excludes_storm_exposure_and_verifies_hashes(tmp_path):
    import json
    import pandas as pd
    import yaml
    from scripts.evaluate_paper_windfields import MODELS, audit, sha256

    checkpoint = tmp_path / 'model.ckpt'
    checkpoint.write_bytes(b'checkpoint')
    source = tmp_path / 'stats.json'
    source.write_text('{}')
    raster = tmp_path / 'raster.tif'
    raster.write_bytes(b'raster')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': {'root': str(tmp_path), 'stats_file': str(source),
                                              'ibtracs_file': str(source)}}))
    run = tmp_path / 'run.json'
    run.write_text(json.dumps({'input_checkpoints': {'resume': None, 'weights_only': None}}))
    registry = {'models': {}, 'files': {str(p): {'sha256': sha256(p)} for p in (config, run)}}
    def row(sid, storm):
        return dict(sample_id=sid, storm_id=storm, eligible=True, is_rapid_intensification=False,
                    path_assets_json=json.dumps({k: sha256(raster) for k in
                                                ('condition_path','context_path','target_path')}),
                    legacy_row_json=json.dumps({k: str(raster) for k in
                                               ('condition_path','context_path','target_path')}))
    for name in MODELS:
        folder = tmp_path / name
        folder.mkdir()
        pd.DataFrame([row('train', 'TRAIN')]).to_csv(folder / 'training-membership.csv', index=False)
        pd.DataFrame([row('val', 'VAL')]).to_csv(folder / 'validation.csv', index=False)
        pd.DataFrame([row('a', 'TRAIN'), row('b', 'VAL'), row('c', 'TEST')]).to_csv(folder / 'test.csv', index=False)
        registry['models'][name] = dict(config=str(config), run_manifest=str(run), dependencies=[],
                                       checkpoint=str(checkpoint), sha256=sha256(checkpoint), data_metadata=[],
                                       stats_file={'sha256': sha256(source)}, ibtracs_file={'sha256': sha256(source)})
    _, cohort = audit(registry, tmp_path, tmp_path)
    assert [r['sample_id'] for r in cohort if r['included']] == ['c']
    assert 'training-membership' in cohort[0]['exclusion_reasons']
    assert 'validation' in cohort[1]['exclusion_reasons']
    # A validation scene from the same storm excludes the entire test storm.
    pd.DataFrame([row('val', 'TEST')]).to_csv(tmp_path / MODELS[0] / 'validation.csv', index=False)
    _, cohort = audit(registry, tmp_path, tmp_path)
    assert not any(r['included'] for r in cohort)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Hash mismatch'):
        audit(registry, tmp_path, tmp_path)
