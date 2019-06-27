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
(defun external-erase-buffer (buffer)
  (erase-buffer
  (string->name buffer)))

(add-act-r-command "erase-buffer" 'external-erase-buffer)

(defun external-run-until-action (action)
(run-until-action (string->name action)))

(add-act-r-command "run-until-action" 'external-run-until-action)


(clear-all)
(require-extra "blending")

(define-model test-blending
    (sgp :sim-hook "similarity_function"
         :cache-sim-hook-results t
         ;:blending-request-hook "new_blend_request"
         :tmp 1.0         ;:seed (1 1) :bll nil :blc 5 :mp 1 :v t :blt t :esc t :ans .25 :rt -5)
         :seed (1 1) :bll nil :blc 5 :mp 1 :v nil :blt nil :esc t :ans nil :rt -5000 :lf 0 :ncnar nil :value->mag second)

  ;(chunk-type observation needsRadio needsFood needsFA needsWater actual)
  ;(chunk-type observation current_altitude heading view_left view_diagonal_left view_center view_diagonal_right view_right)
  ;(chunk-type observation current_altitude view_left view_diagonal_left view_center view_diagonal_right view_right)
  (chunk-type observation hiker_left hiker_diagonal_left hiker_center hiker_diagonal_right hiker_right
              ego_left ego_diagonal_left ego_center ego_diagonal_right ego_right altitude fc)
              ;distance_to_hiker altitude)
  ;(chunk-type decision needsRadio needsFood needsFA needsWater radio food firstaid water)
  ;(chunk-type decision current_altitude heading view_left view_diagonal_left view_center view_diagonal_right view_right
  ;            action)
  ;(chunk-type decision current_altitude view_left view_diagonal_left view_center view_diagonal_right view_right)
  (chunk-type decision hiker_left hiker_diagonal_left hiker_center hiker_diagonal_right hiker_right
              ego_left ego_diagonal_left ego_center ego_diagonal_right ego_right
              distance_to_hiker altitude type
              left_down diagonal_left_down center_down diagonal_right_down right_down
              left_level diagonal_left_level center_level diagonal_right_level right_level
              left_up diagonal_left_up center_up diagonal_right_up right_up fc)
  ;(run-full-time 3600 t)
  ;(schedule-event 3600 (lambda ())); dummy function
  ;(mp-real-time-management :time-function "ticker")

  (p p1
     =imaginal>
       ;altitude =CA
       distance_to_hiker =DTH
       ego_right =ER
       ego_diagonal_right =EDR
       ego_center =EC
       ego_diagonal_left =EDL
       ego_left =EL
       hiker_right =HR
       hiker_diagonal_right =HDR
       hiker_center =HC
       hiker_diagonal_left =HDL
       hiker_left =HL
       fc =FC
     ?blending>
       state free
       buffer empty
       error nil
     ==>
     @imaginal>
     +blending>
       isa decision
       ;altitude =CA
       distance_to_hiker =DTH
       ego_right =ER
       ego_diagonal_right =EDR
       ego_center =EC
       ego_diagonal_left =EDL
       ego_left =EL
       hiker_right =HR
       hiker_diagonal_right =HDR
       hiker_center =HC
       hiker_diagonal_left =HDL
       hiker_left =HL
       fc =FC
       type NAV
       :ignore-slots (altitude distance_to_hiker hiker_left hiker_diagonal_left hiker_center hiker_diagonal_right hiker_right ego_left ego_diagonal_left ego_center ego_diagonal_right ego_right))



;  (p p2
;     =imaginal>
;       wait false
;     =blending>
;       isa decision
;       left_up =LUP
;       diagonal_left_up =DLUP
;       ;action =action
;     ?blending>
;       state free
;     ==>
;     ;!output! (blended value is =val)
;
;     ; Overwrite the blended chunk to erase it and keep it
;     ; from being added to dm.  Not necessary, but keeps the
;     ; examples simpler.
;     =imaginal>
;       wait  true
;     @blending>
;
;     ;;+blending>
;     ;;  isa target
;     ;;  key key-2)
;     )
;  (p p3
;     =imaginal>
;       wait true
;     ==>
;     
;       )

 )

