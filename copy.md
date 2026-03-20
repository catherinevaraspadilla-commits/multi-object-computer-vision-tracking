# Crear zip con todo lo importante
cd ~/multi-object-computer-vision-tracking
zip -r sleap_results_50ep.zip \
  results_50ep/ \
  models/bottomup_ratones_50ep/training_log.csv \
  sleap_22585126.log \
  config_bottomup_ratones_50ep.yaml
Después en tu terminal local (PowerShell o CMD):


scp s4948012@bunya.rcc.uq.edu.au:~/multi-object-computer-vision-tracking/sleap_results_50ep.zip ^
  "C:\Users\lucer\Downloads\multi-animal-pose-tracking\"