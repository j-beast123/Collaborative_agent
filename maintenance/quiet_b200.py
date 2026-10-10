"""Console-only B200 entrypoint; sampling and training identities stay unchanged."""
import logging
import os
from pathlib import Path
import runpy


LOGGER = 'transformers.generation.configuration_utils'
TOP_K_WARNING = (
    "The following generation flags are not valid and may be ignored: ['top_k']. "
    'Set `TRANSFORMERS_VERBOSITY=info` for more details.'
)


class TopKNoticeFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == LOGGER and record.levelno == logging.WARNING
                    and record.getMessage().strip() == TOP_K_WARNING)


def main():
    # RowSampler implements sampling itself; do not change do_sample/top_k
    # merely to silence HF's redundant flag notice. Keep every other warning.
    if os.environ.get('SHERPA_SHOW_TOP_K_WARNING') != '1':
        logging.getLogger(LOGGER).addFilter(TopKNoticeFilter())
    entry = Path(__file__).resolve().parents[1] / 'scripts/online_rl/run_b200.py'
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
