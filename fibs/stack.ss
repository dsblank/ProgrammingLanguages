;; stack representation

;;----------------------------------------------------------------------
;; list implementation

(define make-list-stack
  (lambda ()
    (let ((contents '())
	  (maxsize 0))
      (lambda msg
	(cond
	  ((eq? (car msg) 'empty?) (null? contents))
	  ((eq? (car msg) 'top) (car contents))
	  ((eq? (car msg) 'push!)
	   (set! contents (cons (cadr msg) contents))
	   (set! maxsize (max (length contents) maxsize))
	   'ok)
	  ((eq? (car msg) 'pop!)
	   (let ((top (car contents)))
	     (set! contents (cdr contents))
	     top))
	  ((eq? (car msg) 'show)
	   (display contents)
	   (newline))
	  ((eq? (car msg) 'get-size) (length contents))
	  ((eq? (car msg) 'get-maxsize) maxsize)
	  ((eq? (car msg) 'reset)
	   (set! contents '())
	   (set! maxsize 0))
	  (else 'error))))))

;;----------------------------------------------------------------------
;; vector implementation

(define make-vector-stack
  (lambda (stack-limit)
    (let ((contents (make-vector stack-limit))
	  (next 0)
	  (maxsize 0))
      (lambda msg
	(cond
	  ((eq? (car msg) 'empty?) (= next 0))
	  ((eq? (car msg) 'top) (vector-ref contents (- next 1)))
	  ((eq? (car msg) 'push!)
	   (vector-set! contents next (cadr msg))
	   (set! next (+ 1 next))
	   (set! maxsize (max next maxsize))
	   'ok)
	  ((eq? (car msg) 'pop!)
	   (let ((top (vector-ref contents (- next 1))))
	     (set! next (- next 1))
	     top))
	  ((eq? (car msg) 'show)
	   (let loop ((i next))
	     (if (>= i 0)
		 (begin
		   (display (vector-ref contents i))
		   (display " ")
		   (loop (- i 1)))
		 (newline))))
	  ((eq? (car msg) 'get-size) next)
	  ((eq? (car msg) 'get-maxsize) maxsize)
	  ((eq? (car msg) 'reset)
	   (set! contents (make-vector stack-limit))
	   (set! next 0)
	   (set! maxsize 0))
	  (else 'error))))))
