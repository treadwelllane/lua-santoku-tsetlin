results.mk:
	wget https://github.com/RoaringBitmap/CRoaring/releases/download/v2.1.0/roaring.c
	wget https://github.com/RoaringBitmap/CRoaring/releases/download/v2.1.0/roaring.h
	echo "LIB_CFLAGS += -I$(CURDIR)" >> results.mk
	touch $@
