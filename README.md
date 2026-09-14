# slopmachine-embedding-laion-clap-text-int8

LAION-CLAP のテキストエンコーダーを INT8 ONNX モデルとして提供する、`slopmachine` 向けの埋め込みモデルです。

## インストール

パッケージは次のインデックスから取得します。

```text
https://m1nt-s0da.github.io/slopmachine-pypi/
```

### uv

```powershell
uv add --index slopmachine=https://m1nt-s0da.github.io/slopmachine-pypi/ slopmachine-embedding-laion-clap-text-int8
```

### pip

```powershell
pip install --extra-index-url https://m1nt-s0da.github.io/slopmachine-pypi/ slopmachine-embedding-laion-clap-text-int8
```

## 使用方法

```python
import asyncio

from slopmachine.embedding.laion_clap_text_int8 import LaionClapTextInt8


async def main() -> None:
    async with LaionClapTextInt8() as model:
        tokens = await model.tokenize(["a dog barking", "rain falling"])
        embeddings = await model.encode(tokens.input_ids, tokens.attention_mask)

    print(embeddings.shape)


asyncio.run(main())
```

`embeddings` は各入力テキストに対応する L2 正規化済みの NumPy 配列です。

## 動作要件

- Python 3.14 以上
- ONNX Runtime が利用可能な環境
