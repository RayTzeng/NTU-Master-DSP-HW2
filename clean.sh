#!/bin/bash

read -p "Are you sure you want to clean the directory? (Y/N) " sure
if [ "$sure" == "Y" ]; then
  dir=(decode exp train viterbi feat)
  set -x
  rm -rf ${dir[@]}
  rm -f log/*
fi
