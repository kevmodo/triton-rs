## Test

### Build

```
cargo build && cp ../target/debug/libtriton_example.so ./tritonserver/backends/example
```

### Start Triton

```
docker run -it --rm \
    -p 8000:8000 \
    -v ./tritonserver/backends:/opt/tritonserver/backends \
    -v ./tritonserver/model-repository/:/model-repo \
    --net=host nvcr.io/nvidia/tritonserver:23.07-py3 tritonserver --model-control-mode=explicit --model-repository=/model-repo --log-verbose=1 --load-model=test
```

### Inference

```
curl http://localhost:8000/v2/models/test/infer \
    -H "Content-Type: application/json" \
    -d '{"inputs": [ {"name":"prompt", "datatype":"BYTES", "shape": [3], "data": ["foo"] } ]}'
```

