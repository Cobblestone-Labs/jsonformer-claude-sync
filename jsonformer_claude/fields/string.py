from jsonformer_claude.fields.base import BaseField


class StrField(BaseField):
    def validate_value(self, val: str) -> str:
        if "enum" not in self.schema:
            return True

        return val in self.schema["enum"]

    def reject_value(self, val: str) -> bool:
        whitespace_shifted = val.lstrip()
        if len(whitespace_shifted) < 1:
            return False

        return whitespace_shifted[0] != '"'

    def get_value(self, stream: str) -> str | None:
        whitespace_shifted_stream = stream.lstrip()
        if len(whitespace_shifted_stream) < 2:
            return None

        if whitespace_shifted_stream[0] == '"' and '"' in whitespace_shifted_stream[1:]:
            split_string = whitespace_shifted_stream.split('"')[1]
            print(f">>> RETURNING SPLIT STRING: {split_string}", flush=True)
            return split_string
        else:
            print(f">>> STREAM NOT COMPLETE: '{whitespace_shifted_stream}'", flush=True)

        return None
