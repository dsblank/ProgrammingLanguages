(define fib
   (lambda (n)
     (cond 
       ((= n 1) 1)
       ((= n 2) 1)
       (else (+ (fib (- n 1)) (fib (- n 2)))))))
