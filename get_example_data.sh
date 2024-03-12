#!/bin/sh

url=https://github.com/mueslo/fplore_example_data/releases/download/v4/example_data.tar.gz
dirname=example_data

mkdir $dirname
wget -qO- $url | tar xvz -C $dirname
