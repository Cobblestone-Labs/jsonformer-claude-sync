import anthropic
from typing import List, Union, Any
from jsonformer_claude.fields.base import BaseField
from jsonformer_claude.fields.bool import BoolField
from jsonformer_claude.fields.integer import IntField
from jsonformer_claude.fields.string import StrField
from termcolor import cprint
import json

FIELDS: dict[str, BaseField] = {
    "number": IntField,
    "boolean": BoolField,
    "string": StrField,
}


class JsonformerClaudeMessages:
    """
    Same as JsonformerClaude, but using the messages API
    so that it can support newer models.
    """

    value: dict[str, Any] = {}
    last_anthropic_response: str | None = None
    last_anthropic_response_finished: bool = False
    last_anthropic_stream = None
    llm_request_count = 0

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        json_schema: dict[str, Any],
        prompt: str,
        debug: bool = False,
        **claude_args,
    ):
        self.json_schema = json_schema
        self.prompt = prompt
        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.anthropic_client = anthropic_client
        self.claude_args = claude_args

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        if self.debug_on:
            if is_prompt:
                # cprint(caller, "green", end=" ")
                # cprint(value, "yellow")
                pass
            else:
                cprint(caller, "green", end=" ")
                cprint(f"'{value}'", "blue")

    def _completion(self, prompt: str):
        self.debug("[completion] hitting anthropic", prompt)
        last_anthropic_completion = ""
        self.last_anthropic_response_finished = False
        _stream = self.anthropic_client.messages.stream(
            # prompt=prompt,
            messages=prompt,
            # stop_sequences=[anthropic.HUMAN_PROMPT],
            # stream=True,
            **self.claude_args,
        )
        self.llm_request_count += 1
        with _stream as stream:
            for response in stream.text_stream:
                self.debug("[response-object]: ", response)
                self.debug(
                    "[last-anthropic-completion-before]", last_anthropic_completion
                )
                last_anthropic_completion = last_anthropic_completion + response
                self.debug(
                    "[last-anthropic-completion-after]", last_anthropic_completion
                )

                self.last_anthropic_response = (
                    prompt[-1]["content"] + last_anthropic_completion
                )
                # assistant_index = self.last_anthropic_response.find(anthropic.AI_PROMPT)
                # if assistant_index > -1:
                #     self.last_anthropic_response = self.strip_json_spaces(
                #         self.last_anthropic_response[
                #             assistant_index + len(anthropic.AI_PROMPT) :
                #         ]
                #     )
                yield self.last_anthropic_response
        self.last_anthropic_response_finished = True

    def completion(self, prompt: str):
        self.last_anthropic_stream = self._completion(prompt)
        return self.last_anthropic_stream

    def prefix_matches(self, progress) -> bool:
        if self.last_anthropic_response is None:
            return False
        response = self.last_anthropic_response
        assert (
            len(progress) < len(response) or not self.last_anthropic_response_finished
        )
        while len(progress) >= len(response):
            self.last_anthropic_stream.__next__()
            response = self.last_anthropic_response

        result = response.startswith(progress)

        if not result:
            self.debug(
                "[prefix_matches]",
                "Claude made a mistake",
            )

            cprint(f'>>> PROGRESS: "{progress}"', "red")
            cprint(f'>>> RESPONSE: "{response}"', "magenta")
        else:
            self.debug("[prefix_matches]", "Claude is correct")

        # self.debug("[prefix_matches]", result)
        return result

    def generate_object(
        self, properties: dict[str, Any], obj: dict[str, Any]
    ) -> dict[str, Any]:
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def validate_ref(self, ref):
        if not ref.startswith("#/"):
            raise ValueError("Ref must start with #/")

    def get_definition_by_ref(self, ref) -> dict:
        self.validate_ref(ref)

        locations = ref.split("/")[1:]
        definition = self.json_schema
        for location in locations:
            definition = definition.get(location)

            if not definition:
                raise ValueError("Improper reference")

        return definition

    def get_stream(self):
        progress = self.get_progress()
        prompt = self.get_prompt()

        self.debug("[debug-progress]", progress)
        self.debug("[debug-progress]", prompt)

        stream = self.last_anthropic_response

        if not self.prefix_matches(progress) or stream is None:
            stream = self.completion(prompt)
        else:
            stream = self.last_anthropic_stream

        return stream

    def generate_value(
        self,
        schema: dict[str, Any],
        obj: Union[dict[str, Any], List[Any]],
        key: Union[str, None] = None,
        retries: int = 0,
    ) -> Any:
        if retries > 5:
            self.debug("[completion] EXCEEDED RETRIES RETURNING NONE", str(retries))
            return None

        schema_type = schema.get("type")

        if schema_type in FIELDS:
            field = FIELDS[schema_type](
                schema=schema,
                obj=obj,
                key=key,
                generation_marker=self.generation_marker,
            )
            field.insert_generation_marker()

            stream = self.get_stream()

            for completion in stream:
                progress = self.get_progress()
                self.debug("[JACK]: PROGRESS", progress)
                completion = completion[len(progress) :]
                self.debug("[completion-about-to-gen-value]", completion)
                field_return = field.generate_value(completion)
                self.debug("[completion-just-genned-value]", field_return)

                if field_return.value_valid:
                    return field_return.value
                elif field_return.value_found:
                    # self.debug("[completion]", "retrying")
                    print(f">>> RETRYING INVALID VALUE FOUND: {field_return.value}")
                    self.completion(self.get_prompt())
                    # Could do things like change temperature here
                    return self.generate_value(
                        schema=schema, obj=obj, key=key, retries=retries + 1
                    )
                elif field_return.value_rejected:
                    # self.debug("[completion]", "retrying")
                    print(f">>> RETRYING REJECTED VALUE: {field_return.value}")
                    self.completion(self.get_prompt())
                    # Could do things like change temperature here
                    return self.generate_value(
                        schema=schema, obj=obj, key=key, retries=retries + 1
                    )

        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)

        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)

            return self.generate_object(schema["properties"], new_obj)

        elif discriminator := schema.get("discriminator"):
            property_name = discriminator["propertyName"]
            mapping = discriminator["mapping"]

            property_name_schema = {"type": "string", "enum": [m for m in mapping]}

            new_obj = {}
            new_obj[property_name] = self.generation_marker

            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)

            property_enum_value = self.generate_value(
                schema=property_name_schema, obj=new_obj, key=property_name
            )
            new_obj[property_name] = property_enum_value

            # new_obj.pop(property_name)

            self.debug("[discriminator]", property_enum_value)

            schema = self.get_definition_by_ref(mapping[property_enum_value])
            self.debug("[discriminator]", schema)
            return self.generate_object(properties=schema["properties"], obj=new_obj)

        elif ref := schema.get("$ref"):
            definition = self.get_definition_by_ref(ref)
            return self.generate_value(
                schema=definition,
                obj=obj,
                key=key,
            )

        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: dict[str, Any], arr: List[Any]) -> List[Any]:
        while True:
            try:
                if self.last_anthropic_response is None:
                    # todo: below is untested since we do not support top level arrays yet
                    stream = self.completion(self.get_prompt())
                    for response in stream:
                        completion = response[len(self.get_progress()) :]
                        if completion and completion[0] == ",":
                            self.last_anthropic_response = completion[1:]
                            break
                else:
                    arr.append(self.generation_marker)
                    progress = self.get_progress()
                    arr.pop()
                    progress = progress.rstrip(",")
                    response = self.last_anthropic_response
                    while len(progress) >= len(response):
                        self.last_anthropic_stream.__next__()
                        response = self.last_anthropic_response
                    next_char = response[len(progress)]
                    if next_char == "]":
                        return arr

                value = self.generate_value(item_schema, arr)
                arr[-1] = value
            except StopIteration as e:
                print(f">>> STOP ITERATION: {arr}")
                raise e

    def strip_json_spaces(self, json_string: str) -> str:
        should_remove_spaces = True

        def is_unescaped_quote(index):
            return json_string[index] == '"' and (
                index < 1 or json_string[index - 1] != "\\"
            )

        index = 0
        while index < len(json_string):
            if is_unescaped_quote(index):
                should_remove_spaces = not should_remove_spaces
            elif json_string[index] in [" ", "\t", "\n"] and should_remove_spaces:
                json_string = json_string[:index] + json_string[index + 1 :]
                continue
            index += 1
        return json_string

    def get_progress(self):
        progress = json.dumps(self.value, separators=(",", ":"))
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")
        return self.strip_json_spaces(progress)

    def get_prompt(self):
        """
        As part of the messages API, the prompt is now a list of message objects.
        """

        messages = [
            {
                "role": "user",
                "content": f"{self.prompt}\nOutput result in the following JSON schema format:\n{json.dumps(self.json_schema)}",
            },
            {"role": "assistant", "content": f"{self.get_progress().rstrip()}"},
        ]

        # template = """{HUMAN}{prompt}\nOutput result in the following JSON schema format:\n{schema}{AI}{progress}"""
        # progress = self.get_progress()
        # prompt = template.format(
        #     prompt=self.prompt,
        #     schema=json.dumps(self.json_schema),
        #     progress=progress,
        #     HUMAN=anthropic.HUMAN_PROMPT,
        #     AI=anthropic.AI_PROMPT,
        # )
        # return prompt.rstrip()

        return messages

    def __call__(self) -> dict[str, Any]:
        self.llm_request_count = 0
        self.value = {}
        generated_data = self.generate_object(
            self.json_schema["properties"], self.value
        )
        return generated_data
