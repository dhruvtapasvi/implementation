#!/usr/bin/env bash

# Test schemas against metaschema
jsonschema -i config/schema/vae.schema.json config/schema/json-schema-draft-04.schema.json
jsonschema -i config/schema/deepDense.schema.json config/schema/json-schema-draft-04.schema.json
jsonschema -i config/schema/convolutionalDeepIntermediate.schema.json config/schema/json-schema-draft-04.schema.json

# Test deep dense model configuration instances against schema
jsonschema -i config/model/deepDense/deepDense_28x28_ENC_512x2_1024_DEC_512x3_LAT_32_bce.json config/schema/deepDense.schema.json
jsonschema -i config/model/deepDense/deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce.json config/schema/deepDense.schema.json
jsonschema -i config/model/deepDense/deepDense_64x64_ENC_1024x6_DEC_1024x6_LAT_32_bce.json config/schema/deepDense.schema.json
jsonschema -i config/model/deepDense/deepDense_96x96_ENC_1024_2048_2048_DEC_2048_2048_1024_LAT_32_bce.json config/schema/deepDense.schema.json

# Test deep dense convolutional model configuration instances against schema
jsonschema -i config/model/convolutionalDeepIntermediate/conv_28x28_3_8_ENC_1024_DEC_1024_LAT_32_bce.json config/schema/convolutionalDeepIntermediate.schema.json
jsonschema -i config/model/convolutionalDeepIntermediate/conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce.json config/schema/convolutionalDeepIntermediate.schema.json
jsonschema -i config/model/convolutionalDeepIntermediate/conv_96x96_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce.json config/schema/convolutionalDeepIntermediate.schema.json
