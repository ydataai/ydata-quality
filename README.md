# ydata-quality
YData open-source lib for Data Quality.

### Contributing
We leverage internal comments to mark notes for future contributions. Use TODO for new functionality, FIXME for bugfixing.
```python
def awesome_function():
    # FIXME: make me work
    # TODO: add new functionality
```

### venv
_How to create a virtual environment to run the ydata_quality package._

```bash
# create the virtual environment
$ python -m venv .venv

# activate the environment
$ . ./.venv/bin/activate

# build the package
$ make package

# install from package
$ make install
```
