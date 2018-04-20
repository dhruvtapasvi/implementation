#!/usr/bin/env bash

# Test schemas against metaschema
jsonschema -i config/schema/vae.schema.json config/schema/json-schema-draft-04.schema.json
jsonschema -i config/schema/convolutional.schema.json config/schema/json-schema-draft-04.schema.json
jsonschema -i config/schema/dense.schema.json config/schema/json-schema-draft-04.schema.json

# Test convolutional vae model configuration instances against schema
jsonschema -i config/model/convolutional/conv_28x28_3_8_256_2_bce.json config/schema/convolutional.schema.json
jsonschema -i config/model/convolutional/conv_64x64_6_32_2048_64_bce.json config/schema/convolutional.schema.json
jsonschema -i config/model/convolutional/conv_64x64_7_8_256_10_bce.json config/schema/convolutional.schema.json
jsonschema -i config/model/convolutional/conv_64x64_7_16_256_32_bce.json config/schema/convolutional.schema.json
jsonschema -i config/model/convolutional/conv_64x64_7_32_512_64_bce.json config/schema/convolutional.schema.json
jsonschema -i config/model/convolutional/conv_96x96_6_8_256_10_bce.json config/schema/convolutional.schema.json
jsonschema -i config/model/convolutional/conv_96x96_6_16_256_10_bce.json config/schema/convolutional.schema.json

# Test dense vae model configuration instances against schema
jsonschema -i config/model/dense/dense_28x28_keras_autoencoders_tutorial.json config/schema/dense.schema.json

# Test deep dense model configuration instances against schema
jsonschema -i config/model/deepDense/deepDense_28x28_ENC_512_512_1024_1024_DEC_512_512_512_512_LAT_512_bce.json config/schema/deepDense.schema.json
