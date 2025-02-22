
flist = $(wildcard cmtf_pls/figures/figure*.py)

all: $(patsubst cmtf_pls/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: cmtf_pls/figures/figure%.py
	mkdir -p output
	rye run python ./cmtf_pls/figures/common.py $*

test:
	rye run pytest -s -v -x

coverage.xml:
	rye run pytest --cov=cmtf_pls --cov-report=xml --cov-config=.github/workflows/coveragerc

testprofile:
	rye run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

clean:
	rm -rf output
