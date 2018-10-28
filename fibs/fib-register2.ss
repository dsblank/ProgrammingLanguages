;; register machine using a continuation stack of frames

(load "cons-counter.ss")
(load "stack.ss")

(define <n> #f)
(define <k> #f)
(define <value> #f)
(define <pc> #f)

(define fib
  (lambda (n)
    ;(counter 'reset)
    ;(set! <k> (make-list-stack))
    (set! <k> (make-vector-stack 100))
    (set! <n> n)
    (<k> 'push! (make-frame0))
    (set! <pc> fib-cps)
    (loop)
    ;(printf "~a cons cells allocated~n" (counter 'value))
    <value>))

(define loop
  (lambda ()
    (if <pc>
	(begin
	  (<pc>)
	  ;;(<k> 'show)
	  (loop)))))

(define fib-cps
  (lambda ()
    (if (< <n> 3)
	(begin
	  (set! <value> 1)
	  (set! <pc> apply-cont))
	(begin
	  (<k> 'push! (make-frame1 <n>))
	  (set! <n> (- <n> 1))
	  (set! <pc> fib-cps)))))

(define make-frame0 (lambda () (lizt 'frame0)))
(define make-frame1 (lambda (n) (lizt 'frame1 n)))
(define make-frame2 (lambda (v1) (lizt 'frame2 v1)))

(define apply-cont
  (lambda ()
    (let ((frame (<k> 'pop!)))
      (cond
        ((eq? (car frame) 'frame0)
	 (set! <pc> #f))
	((eq? (car frame) 'frame1)
	 (set! <n> (- (cadr frame) 2))
	 (<k> 'push! (make-frame2 <value>))
	 (set! <pc> fib-cps))
	((eq? (car frame) 'frame2)
	 (set! <value> (+ (cadr frame) <value>))
	 (set! <pc> apply-cont))
	(else 'error)))))
