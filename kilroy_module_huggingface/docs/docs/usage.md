# Usage

This package provides an interface to any HuggingFace model
that complies with the **kilroy** module API.

## Prerequisites

By default, the module uses the `gpt2` model.
If you want to use another model,
you need to pick a sufficiently powerful one for text generation.
You can find a list of models
[here](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

Also make sure that you have enough memory to load the model.

## Running the server

To run the module server, install the package and run the following command:

```sh
kilroy-module-huggingface
```

This will start the face server on port 11000 by default.
Then you can communicate with the server, for example by using
[this package](https://github.com/kilroybot/kilroy-module-client-py-sdk).
