.PHONY: download preprocess train evaluate export stream clean

download:
	python -m src.data.download

preprocess:
	python -m src.data.preprocess

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml

export:
	python scripts/export_tableau.py --config configs/default.yaml

stream:
	python scripts/stream_replay.py --config configs/default.yaml --mode fast

stream-realtime:
	python scripts/stream_replay.py --config configs/default.yaml --mode realtime --interval 5

clean:
	rm -rf data/processed/* outputs/checkpoints/* outputs/figures/* outputs/tableau/* outputs/streaming/*
