#!/usr/bin/env bash

jsonschema -i config/model/vae/vae.schema.json config/json-schema-draft-04.schema.json
jsonschema -i config/model/convolutional/convolutional.schema.json config/json-schema-draft-04.schema.json

