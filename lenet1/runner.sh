#!/bin/bash
for ((i=0; i < 10000; i++ )); do
  result=$(./out/test_$i.out)
  read should is <<<${result//[^0-9]/ }
  echo $is
done
