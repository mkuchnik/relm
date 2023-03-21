"""Read The Pile Data and convert to txt and print.

Modified parser utilities from:
https://github.com/EleutherAI/the-pile/blob/master/the_pile/tfds_pile.py
"""
import io

import jsonlines
import simdjson as json
import zstandard

parser = json.Parser()


def json_parser(x):
    """Try json parsing."""
    try:
        line = parser.parse(x).as_dict()
        return line
    except ValueError:
        return x


class PileReader:
    """Read The Pile data."""

    def __init__(self, filenames, para_joiner='\n\n'):
        """Create a reader from filenames."""
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.para_joiner = para_joiner

    def _read_fn(self, filename):
        """Read and decompress the file.

        Schema:
        result["text"] = <lines>
        """
        with open(filename, "rb") as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream, loads=json_parser)
            for item in reader:
                result = dict()
                if isinstance(item, str):
                    result['text'] = item
                else:
                    text = item['text']
                    if isinstance(text, list):
                        text = self.para_joiner.join(text)
                    result['text'] = text
                yield result

    def __iter__(self):
        """Iterate the reader from filenames."""
        for filename in self.filenames:
            return self._read_fn(filename)


if __name__ == "__main__":
    filename = "00.jsonl.zst"
    reader = PileReader(filename)
    for x in reader:
        print(x["text"])
