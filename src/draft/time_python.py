import time

start = time.time()
second = time.time() - start

minute, second = divmod(234322.34343, 60)
hour, minute = divmod(minute, 60)

print('[{}:{}:{}]'.format(int(hour), int(minute), int(second)))
