#!/bin/sh

mkdir example_data
wget -qO- https://mueslo.de/fplore/example_data.tar.gz | tar xvz -C example_data
