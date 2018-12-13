#! /bin/sh
echo "*** Training kenlm on $2 ***"
echo "*** Running $1 -o $2 < $3 > $3.arpa"
$1 -o $2 < $3 > $3.arpa
echo "*** Writing output to  $3.arpa ***"
