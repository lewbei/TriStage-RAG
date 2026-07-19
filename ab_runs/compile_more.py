import sys, py_compile
files = [
 r'tristage_rag/retrieval_pipeline.py',
 r'tristage_rag/stage2_rescorer.py',
 r'tristage_rag/stage3_reranker.py',
 r'run_benchmark.py',
]
ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print('OK:', f)
    except Exception as e:
        ok = False
        print('ERROR:', f, e)
sys.exit(0 if ok else 1)
