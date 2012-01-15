all: index.html

index.html: index.md
	pandoc --offline -s -t slidy -o $@ $<

clean:
	-rm -f index.html
