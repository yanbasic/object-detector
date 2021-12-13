import concurrent.futures
from time import sleep
def foo(bar):
    print('hello {}'.format(bar))
    sleep(5)
    return 'foo'

with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(foo, 'world!')
    sleep(3)
    print('111')
    return_value = future.result()
    print(return_value)