from pathlib import Path
import re

N = 10000

with open("all_tests.c", "w") as out:
    out.write('#include "all_tests.h"\n\n')
    out.write("param_t TEST_IMAGES[NUM_TESTS][IMAGE_SIZE] = {\n")

    for i in range(N):
        text = Path(f"inputs/test_{i}.h").read_text()

        # Extract everything inside { ... }
        data = re.search(r'\{(.*)\}', text, re.S).group(1)

        out.write("{")
        out.write(data.strip())
        out.write("}")

        if i != N - 1:
            out.write(",")

        out.write("\n")

    out.write("};\n")
    out.write("size_p TEST_LABELS[NUM_TESTS] = {")
    for i in range(N):
        text = Path(f"inputs/test_{i}.h").read_text()
        label = re.search(r'LABEL_TEST_\d+ (\d*)\n', text, re.S).group(1)
        out.write(f"{label}")
        if i < N - 1:
            out.write(", ")
    out.write("};\n")
