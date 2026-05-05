## BF16 sandbox

We **optionally replace BLIS `sgemm`** with a path that **packs float32 A/B to BF16** and runs a **BF16 microkernel** (accumulating into float32 C).

### Build + run
First clone the flame/blis repository. Then place this directory in the sandbox directory. Then rename it from "blis-bf16" to just "bf16"

Then From the BLIS top-level directory:

```bash
./configure -s bf16 auto
make -j
make -C sandbox/bf16 driver
./run_sgemm_bf16 128 128 128
```

### Enable it for real `sgemm` calls

```bash
export BLIS_SANDBOX_BF16=1
```

