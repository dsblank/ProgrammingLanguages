;; register machine

(load "cons-counter.ss")

(define <n> #f)
(define <k> #f)
(define <value> #f)
(define <pc> #f)

(define fib
  (lambda (n)
    ;(counter 'reset)
    (set! <n> n)
    (set! <k> (make-cont0))
    (set! <pc> fib-cps)
    (loop)
    ;(printf "~a cons cells allocated~n" (counter 'value))
    <value>))

(define loop
  (lambda ()
    (if <pc>
	(begin
	  (<pc>)
	  ;(display <k>) (newline)
	  (loop)))))

(define fib-cps
  (lambda () ;; <n> <k>
    (if (< <n> 3)
	(begin
	  (set! <value> 1)
	  (set! <pc> apply-cont))
	(begin
	  (set! <k> (make-cont1 <n> <k>))
	  (set! <n> (- <n> 1))
	  (set! <pc> fib-cps)))))

;;- - - - - - - - - - - - - - - - - - - - - - - - - - - -
;; continuations represented as nested lists

(define make-cont0 (lambda () (lizt 'cont0)))
(define make-cont1 (lambda (n k) (lizt 'cont1 n k)))
(define make-cont2 (lambda (v1 k) (lizt 'cont2 v1 k)))

(define apply-cont
  (lambda () ;; <k> <value>
    (cond
      ((eq? (car <k>) 'cont0)
       (set! <pc> #f))
      ((eq? (car <k>) 'cont1)
       (set! <n> (- (cadr <k>) 2))
       (set! <k> (make-cont2 <value> (caddr <k>)))
       (set! <pc> fib-cps))
      ((eq? (car <k>) 'cont2)
       (set! <value> (+ (cadr <k>) <value>))
       (set! <k> (caddr <k>))
       (set! <pc> apply-cont))
      (else 'error))))

;;- - - - - - - - - - - - - - - - - - - - - - - - - - - -
;; continuations represented as lists of frames

;;(define make-cont0 (lambda () (conz (lizt 'frame0) '())))
;;(define make-cont1 (lambda (n k) (conz (lizt 'frame1 n) k)))
;;(define make-cont2 (lambda (v1 k) (conz (lizt 'frame2 v1) k)))

;;(define apply-cont
;;  (lambda () ;; <k> <value>
;;    (let ((frame (car <k>)))
;;      (cond
;;        ((eq? (car frame) 'frame0)
;;	 (set! <pc> #f))
;;	((eq? (car frame) 'frame1)
;;	 (set! <n> (- (cadr frame) 2))
;;	 (set! <k> (make-cont2 <value> (cdr <k>)))
;;	 (set! <pc> fib-cps))
;;	((eq? (car frame) 'frame2)
;;	 (set! <value> (+ (cadr frame) <value>))
;;	 (set! <k> (cdr <k>))
;;	 (set! <pc> apply-cont))
;;	(else 'error)))))
