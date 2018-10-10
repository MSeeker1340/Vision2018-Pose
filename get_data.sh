# get COCO dataset
mkdir dataset
mkdir dataset/COCO/
mkdir dataset/COCO/images
mkdir dataset/COCO/images/mask2014
mkdir dataset/COCO/mat
mkdir dataset/COCO/json

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip

echo "Unzipping files ..."
unzip person_keypoints_trainval2014.zip -d dataset/COCO/ > /dev/null
unzip val2014.zip -d dataset/COCO/images > /dev/null
unzip test2014.zip -d dataset/COCO/images > /dev/null
unzip train2014.zip -d dataset/COCO/images > /dev/null
unzip test2015.zip -d dataset/COCO/images > /dev/null

rm -f *.zip
