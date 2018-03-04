#!/usr/bin/env bash

# Test schemas against metaschema
jsonschema -i config/schema/vae.schema.json config/schema/json-schema-draft-04.schema.json
jsonschema -i config/schema/convolutional.schema.json config/schema/json-schema-draft-04.schema.json

# Test convolutional vae model configuration instances against schema
jsonschema -i config/model/convolutional/mnist_conv_bce_2.json config/schema/convolutional.schema.json
