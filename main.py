import mnist_train
import mnist_eval
import _thread

def main(argv = None):
    _thread.start_new_thread(mnist_eval.main(), ())
    _thread.start_new_thread(mnist_train.main(),())


if __name__ == '__main__':
    main()