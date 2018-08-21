;; Simple model to show blending in action.
;; Load the model and call run to see the 
;; trace of two blended retrievals.  The
;; second one will usually fail because
;; it will be below the retrieval threshold.
;;
;; This is a simple case without partial
;; matching enabled and demonstrating both
;; a numeric value blending and a chunk based 
;; blending.


(clear-all)
(require-extra "blending")

(define-model test-blending
    (sgp :sim-hook "similarity_function"
         ;:blending-request-hook "new_blend_request"
         :tmp 1.0
         ;:seed (1 1) :bll nil :blc 5 :mp 1 :v t :blt t :esc t :ans .25 :rt -5)
         :seed (1 1) :bll nil :blc 5 :mp 1 :v t :blt t :esc t :ans nil :rt -5 :value->mag second)

  (chunk-type observation one two three four five six seven eight nine delta_x delta_y)
  ;(chunk-type observation needsFood needsWater actual)
  (chunk-type decision one two three four five six seven eight nine delta_x delta_y actual_x actual_y)
  ;(chunk-type decision needsFood needsWater food water)
  (chunk-type target key value size)
  (chunk-type size (size-type t))
  
  ;; some chunks which don't need to be in DM
  (define-chunks 
      (key-1 isa chunk)
      (key-2 isa chunk))

  
  ;; Here are the chunks for the blending test
  

  
  ;; Provide the similarities between the sizes
  ;; because blending will use that even though 
  ;; partial matching is not enabled.
  

  
  ;; Very simple model
  ;; Make a blending request for a target chunk
  ;; with key value key-1 and if such a chunk
  ;; is found make a blending request for a
  ;; target chunk with a key value of key-2
  
  (p p1
     =imaginal>
       one =one
       two =two

     ?blending>
       state free
       buffer empty
       error nil
     ==>
     ;@imaginal>
     +blending>
       isa decision
       one =one
       two =two

       actual_x nil
       actual_y nil)

  
  (p p2
     =blending>
       isa decision
       one =one
       two =two
       delta_x =dx
       delta_y =dy
       actual_x =actual_x
       actual_y =actual_y

     ?blending>
       state free
     ==>
     ;!output! (blended value is =val)
     
     ; Overwrite the blended chunk to erase it and keep it 
     ; from being added to dm.  Not necessary, but keeps the 
     ; examples simpler.
     
     ;@blending>    
     
     ;;+blending>
     ;;  isa target
     ;;  key key-2)
     )
  )

