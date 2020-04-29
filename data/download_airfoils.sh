#!/bin/bash

echo downloading...
wget http://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.tar.gz

echo unpacking...
tar xzf ./coord_seligFmt.tar.gz 

mkdir ./airfoil_database
mkdir ./airfoil_database_test

cd ./coord_seligFmt/

# TODO fix instead of removing
# cleanup:
# remove airfoils with text comments
rm ag24.dat ag25.dat ag26.dat ag27.dat nasasc2-0714.dat goe795sm.dat naca1.dat
# remove some non ascii ones
rm goe187.dat goe188.dat goe235.dat

# move only selected files to make sure we have the right sets
echo moving...

# test set
mv ag09.dat ah63k127.dat ah94156.dat bw3.dat clarym18.dat   ../airfoil_database_test/ 
mv e221.dat e342.dat e473.dat e59.dat e598.dat              ../airfoil_database_test/ 
mv e864.dat fx66h80.dat fx75141.dat fx84w097.dat goe07k.dat ../airfoil_database_test/ 
mv goe147.dat goe265.dat goe331.dat goe398.dat goe439.dat   ../airfoil_database_test/ 
mv goe501.dat goe566.dat goe626.dat goe775.dat hq1511.dat   ../airfoil_database_test/ 
mv kc135d.dat m17.dat mh49.dat mue139.dat n64012a.dat       ../airfoil_database_test/ 

# training set
mv *.dat ../airfoil_database/

cd ..
rm -fr ./coord_seligFmt/
rm ./coord_seligFmt.tar.gz 

echo done!
exit 1

