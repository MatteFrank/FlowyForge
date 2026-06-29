from __future__ import annotations

import math

import numpy as np

from flowyforge.data_plugins.collide_v2.hf_collide1m import summarize_hf_event_row


def test_summarize_hf_event_row_converts_lists_to_scalar_features() -> None:
    row = {
        "FullReco_MET_MET": [10.0, 20.0, 30.0],
        "FullReco_MET_Phi": np.asarray([1.0, 3.0]),
        "FullReco_JetAK4_PT": [[5.0, 10.0], [15.0]],
    }

    summary = summarize_hf_event_row(
        row,
        columns=["FullReco_MET_MET", "FullReco_MET_Phi", "FullReco_JetAK4_PT"],
        process_name="WJetsToLNu_13TeV-madgraphMLM-pythia8",
        process_id=123,
        event_id=7,
    )

    assert summary["event_id"] == 7
    assert summary["process_name"] == "WJetsToLNu_13TeV-madgraphMLM-pythia8"
    assert summary["process_id"] == 123
    assert summary["FullReco_MET_MET_len"] == 3
    assert summary["FullReco_MET_MET_mean"] == 20.0
    assert math.isclose(summary["FullReco_MET_MET_std"], np.std([10.0, 20.0, 30.0]))
    assert summary["FullReco_MET_MET_min"] == 10.0
    assert summary["FullReco_MET_MET_max"] == 30.0
    assert summary["FullReco_MET_MET_sum"] == 60.0
    assert summary["FullReco_MET_Phi_len"] == 2
    assert summary["FullReco_MET_Phi_mean"] == 2.0
    assert summary["FullReco_JetAK4_PT_len"] == 3
    assert summary["FullReco_JetAK4_PT_sum"] == 30.0


def test_summarize_hf_event_row_handles_empty_missing_and_nonnumeric_values() -> None:
    row = {
        "FullReco_MET_MET": [],
        "FullReco_MET_Phi": None,
        "L1T_MET_MET": ["not", "numeric"],
    }

    summary = summarize_hf_event_row(
        row,
        columns=["FullReco_MET_MET", "FullReco_MET_Phi", "L1T_MET_MET", "MissingColumn"],
    )

    for column in ["FullReco_MET_MET", "FullReco_MET_Phi", "L1T_MET_MET", "MissingColumn"]:
        assert summary[f"{column}_len"] == 0
        assert summary[f"{column}_mean"] == 0.0
        assert summary[f"{column}_std"] == 0.0
        assert summary[f"{column}_min"] == 0.0
        assert summary[f"{column}_max"] == 0.0
        assert summary[f"{column}_sum"] == 0.0
