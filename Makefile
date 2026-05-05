## Makefile for the bf16 sandbox driver.
##
## Usage (from BLIS top-level):
##   make -C sandbox/bf16 driver
##   ./run_sgemm_bf16 128 128 128
##
## Assumes BLIS was built already with:
##   ./configure -s bf16 auto && make -j

BLIS_TOP ?= ../..
CONF     ?= haswell

CC       ?= cc
CFLAGS   ?= -O2

DRIVER   := $(BLIS_TOP)/run_sgemm_bf16

.PHONY: driver clean

driver:
	$(CC) $(CFLAGS) \
	  -I$(BLIS_TOP)/include/$(CONF) -I$(BLIS_TOP)/frame/include \
	  $(BLIS_TOP)/sandbox/bf16/run_sgemm_bf16.c \
	  $(BLIS_TOP)/lib/$(CONF)/libblis.a -lm -lpthread \
	  -o $(DRIVER)

clean:
	rm -f $(DRIVER)

