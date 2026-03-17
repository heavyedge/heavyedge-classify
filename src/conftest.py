import os

from heavyedge_classify.samples import clean_all_samples, make_all_samples

if int(os.getenv("HEAVYEDGE_TEST_REBUILD", 1)):
    print("\nRe-building package data...")
    clean_all_samples()
    make_all_samples(progress=True)
    print("Finished re-building.")
