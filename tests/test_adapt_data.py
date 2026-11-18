import json

import pytest

from zubaan.adapt.data import (
    BlockSampler,
    PackedBlocks,
    build_english_replay,
    clean_english_line,
    encode_lines,
    english_block_count,
    find_overlaps,
    load_blocks,
    mix_plan,
    pack_documents,
    prepare_blocks,
    require_no_overlap,
)

EOS = 99


def test_pack_documents_worked_example():
    # Docs of 3, 2 and 4 tokens: stream = (1+3) + (1+2) + (1+4) = 12 ids, block size 5 -> 2 blocks
    # and 2 dropped ids.
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    blocks, stats = pack_documents(docs, 5, EOS)
    assert len(blocks) == 2
    assert blocks.block(0) == [EOS, 1, 2, 3, EOS]
    assert blocks.block(1) == [4, 5, EOS, 6, 7]
    assert (stats.n_docs, stats.n_tokens, stats.n_blocks, stats.dropped_tokens) == (3, 12, 2, 2)


def test_pack_documents_skips_empty_docs_and_counts_leftovers():
    blocks, stats = pack_documents([[], [1], []], 4, EOS)
    assert len(blocks) == 0
    assert stats.n_docs == 1 and stats.dropped_tokens == 2


def test_packed_blocks_roundtrip_on_disk(tmp_path):
    blocks, _ = pack_documents([[1, 2, 3, 4, 5, 6, 7]] * 3, 4, EOS)
    path = tmp_path / "b.u32"
    blocks.save(path)
    assert path.stat().st_size == len(blocks) * 4 * 4
    again = PackedBlocks.load(path, 4)
    assert [again.block(i) for i in range(len(again))] == [
        blocks.block(i) for i in range(len(blocks))
    ]
    with pytest.raises(IndexError):
        again.block(len(again))


def test_packed_blocks_reject_bad_shapes(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        PackedBlocks(1)
    (tmp_path / "x").write_bytes(b"\x00" * 6)
    with pytest.raises(ValueError, match="4-byte"):
        PackedBlocks.load(tmp_path / "x", 2)


def test_encode_lines_batches_and_keeps_order():
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return [[len(t)] for t in texts]

    out = list(encode_lines(["a", "bb", "ccc", "dddd", "eeeee"], enc, batch=2))
    assert out == [[1], [2], [3], [4], [5]]
    assert calls == [2, 2, 1]


def test_english_block_count_worked_example():
    assert english_block_count(90, 0.1) == 10  # 10 of 100 blocks are English
    assert english_block_count(100, 0.0) == 0
    with pytest.raises(ValueError):
        english_block_count(10, 1.0)


def test_mix_plan_has_every_urdu_block_and_the_right_english_share():
    plan = mix_plan(90, 50, 0.1, seed=3)
    assert len(plan) == 100
    assert sorted(i for s, i in plan if s == 0) == list(range(90))
    en = [i for s, i in plan if s == 1]
    assert len(en) == 10 and len(set(en)) == 10  # without replacement


def test_mix_plan_is_deterministic_and_seed_changes_it():
    assert mix_plan(40, 20, 0.2, 1) == mix_plan(40, 20, 0.2, 1)
    assert mix_plan(40, 20, 0.2, 1) != mix_plan(40, 20, 0.2, 2)


def test_mix_plan_raises_when_english_is_short():
    with pytest.raises(ValueError, match="needs 10 blocks but only 3 exist"):
        mix_plan(90, 3, 0.1, 0)


def _blocks(first_id: int, n: int, size: int = 4) -> PackedBlocks:
    flat = []
    for b in range(n):
        flat += [first_id + b] * size
    from array import array

    return PackedBlocks(size, array("I", flat))


def test_sampler_resume_reproduces_the_stream_across_epochs():
    ur, en = _blocks(1000, 9), _blocks(2000, 5)
    s = BlockSampler(ur, en, 0.1, seed=7)
    assert s.epoch_length == 10
    full = s.blocks_at(0, 25)  # 2.5 epochs
    again = BlockSampler(ur, en, 0.1, seed=7)
    assert again.blocks_at(0, 25) == full
    # Resuming at any position gives the same continuation, including across the epoch edge.
    assert BlockSampler(ur, en, 0.1, 7).blocks_at(8, 6) == full[8:14]
    # Each epoch holds every Urdu block exactly once.
    for e in range(2):
        epoch = s.blocks_at(e * 10, 10)
        assert sorted(b[0] for b in epoch if b[0] < 2000) == list(range(1000, 1009))
    # A later epoch shuffles differently.
    assert full[:10] != full[10:20]


def test_sampler_without_english_and_with_too_little():
    ur = _blocks(1, 4)
    s = BlockSampler(ur, None, 0.5, seed=0)
    assert s.epoch_length == 4  # no English given, so no replay
    with pytest.raises(ValueError, match="not enough English"):
        BlockSampler(ur, _blocks(100, 1), 0.5, seed=0)
    with pytest.raises(ValueError, match="same block size"):
        BlockSampler(ur, _blocks(100, 9, size=8), 0.1, seed=0)


def test_overlap_guard_finds_a_copied_sentence():
    flores = ["اس کتاب میں بہت سی دلچسپ باتیں لکھی ہوئی ہیں جو سب کو پسند آئیں گی"]
    lines = [
        "یہ ایک بالکل مختلف جملہ ہے جس میں کوئی مشترک الفاظ نہیں ہیں",
        "آج ہم نے کہا کہ اس کتاب میں بہت سی دلچسپ باتیں لکھی ہوئی ہیں اور بس",
    ]
    assert find_overlaps(lines, flores) == [1]
    with pytest.raises(ValueError, match="1 Urdu training lines overlap"):
        require_no_overlap(lines, flores, "Urdu")
    assert require_no_overlap(lines[:1], flores, "Urdu") == 1


def test_overlap_guard_refuses_to_run_without_eval_texts():
    with pytest.raises(ValueError, match="would do nothing"):
        find_overlaps(["abc def"], [])


def test_clean_english_line():
    long = " ".join(["word"] * 40)
    assert clean_english_line(long) == long
    assert clean_english_line("Short heading") is None
    assert clean_english_line(" ".join(["слово"] * 40)) is None  # not ASCII


def test_build_english_replay_filters_dedups_and_guards(tmp_path):
    good = " ".join(f"alpha{i}" for i in range(40))
    copied = "the quick brown fox jumps over the lazy dog " * 6
    article = "\n".join(["Heading", good, good, copied])
    meta = build_english_replay(
        [article], tmp_path / "en.txt", eval_texts=[copied.strip()], max_mb=1.0
    )
    assert (tmp_path / "en.txt").read_text(encoding="utf-8") == good + "\n"
    assert meta["lines"] == 1
    assert meta["dropped_duplicate"] == 1 and meta["dropped_eval_overlap"] == 1
    assert meta["dropped_filter"] == 1


def test_prepare_blocks_writes_files_and_metadata(tmp_path):
    def enc(texts):
        return [[ord(c) % 50 + 1 for c in t] for t in texts]

    urdu = ["کتاب پڑھنا " * 6] * 20
    english = ["reading books daily " * 6] * 10
    meta = prepare_blocks(
        urdu,
        english,
        enc,
        eos_id=0,
        out_dir=tmp_path,
        block_size=16,
        urdu_eval_texts=["بالکل مختلف متن جو کہیں نہیں آتا اس میں کچھ بھی نہیں ہے"],
        english_eval_texts=["something else entirely unrelated to the training text here"],
    )
    saved = json.loads((tmp_path / "blocks_meta.json").read_text(encoding="utf-8"))
    assert saved == meta
    assert meta["overlap_guard"]["hits"] == 0
    ur, en, loaded = load_blocks(tmp_path)
    assert len(ur) == meta["urdu"]["n_blocks"] > 0
    assert len(en) == meta["english"]["n_blocks"] > 0
    assert loaded["block_size"] == 16
    assert ur.block(0)[0] == 0  # every block stream starts with the end-of-text id


def test_prepare_blocks_stops_before_writing_when_a_line_leaks(tmp_path):
    leak = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    with pytest.raises(ValueError, match="overlap"):
        prepare_blocks(
            ["اردو متن " * 8],
            [leak],
            lambda ts: [[1] for _ in ts],
            eos_id=0,
            out_dir=tmp_path,
            block_size=4,
            urdu_eval_texts=["اردو متن " * 8],
            english_eval_texts=[leak],
        )
    assert not list(tmp_path.iterdir())


def test_prepare_blocks_can_drop_leaking_lines_and_counts_them(tmp_path):
    leak = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    fine = "one two three four five six seven eight nine ten eleven"
    meta = prepare_blocks(
        ["اردو متن " * 8],
        [leak, fine, fine + " twelve"],
        lambda ts: [[7] * len(t.split()) for t in ts],
        eos_id=0,
        out_dir=tmp_path,
        block_size=4,
        urdu_eval_texts=["بالکل مختلف متن جو کہیں نہیں آتا اس میں کچھ بھی نہیں ہے"],
        english_eval_texts=[leak],
        drop_overlaps=True,
    )
    g = meta["overlap_guard"]
    assert (g["english_lines_checked"], g["english_lines_dropped"], g["hits"]) == (3, 1, 0)
    assert g["urdu_lines_dropped"] == 0
    # Two English lines of 11 and 12 words, plus one end-of-text id each, are packed.
    assert meta["english"]["n_docs"] == 2 and meta["english"]["n_tokens"] == 25
