;; provides the functions conz and lizt, which keep track of the total
;; number of cons cells allocated

(define make-counter
  (lambda ()
    (let ((count 0))
      (lambda msg
	(cond
	  ((eq? (car msg) 'reset) (set! count 0) 'ok)
	  ((eq? (car msg) 'value) count)
	  ((eq? (car msg) 'add) (set! count (+ (cadr msg) count)) 'ok)
	  (else 'error))))))

(define counter (make-counter))

(define lizt
  (lambda args
    (counter 'add (length args))
    (apply list args)))

(define conz
  (lambda args
    (counter 'add 1)
    (apply cons args)))
