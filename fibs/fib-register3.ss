;; register machine using an unframed continuation stack

(load "cons-counter.ss")
(load "stack.ss")

(define <n> #f)
(define <k> #f)
(define <value> #f)
(define <pc> #f)

(define fib
  (lambda (n)
    ;;(counter 'reset)
    ;(set! <k> (make-list-stack))
    (set! <k> (make-vector-stack 100))
    (set! <n> n)
    (<k> 'push! 'frame0)
    (set! <pc> fib-cps)
    (loop)
    ;;(printf "~a cons cells allocated~n" (counter 'value))
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
	  (<k> 'push! <n>)
	  (<k> 'push! 'frame1)
	  (set! <n> (- <n> 1))
	  (set! <pc> fib-cps)))))

(define apply-cont
  (lambda ()
    (let ((frame-tag (<k> 'pop!)))
      (cond
        ((eq? frame-tag 'frame0)
	 (set! <pc> #f))
	((eq? frame-tag 'frame1)
	 (let ((n (<k> 'pop!)))
	   (set! <n> (- n 2)))
	 (<k> 'push! <value>)
	 (<k> 'push! 'frame2)
	 (set! <pc> fib-cps))
	((eq? frame-tag 'frame2)
	 (let ((v1 (<k> 'pop!)))
	   (set! <value> (+ v1 <value>)))
	 (set! <pc> apply-cont))
	(else 'error)))))
