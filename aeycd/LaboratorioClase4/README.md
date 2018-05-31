# Curación de Datos de Permisos de Construcción en San Francisco

Se trabajó con un dataset de Permisos de Construcción de Edificios en San Francisco (https://www.kaggle.com/aparnashastry/building-permit-applications-data/).

El dataset se obtuvo [aquí](https://www.kaggle.com/aparnashastry/building-permit-applications-data/data) y se guarda localmente en `data/Building_Permits.csv`, junto con el archivo de descripción de las columnas `data/DataDictionaryBuildingPermit.xlsx`

Para el proceso de curación se sigió el procedimiento descripto en el *notebook* `Laboratorio Clase 4.ipynb`, el cual se puede abrir y ejecutar luego de correr el contenedor de Docker para jupyter notebook:

	```console
	docker run -it --rm -v $PWD:/home/jovyan/work -e NB_UID=`id -u` -e NB_GID=`id -u` -p 8888:8888 --user root jupyter/scipy-notebook
	```

El dataset ya procesado se encuentra en `data/Building_Permits_final.csv` y la descripción final de las variables se puede ver en la siguiente tabla:


Variable | Descripción | Tipo 
--- |---|---
Record_ID | Id del registro | Numeric
Permit_Number | Numero asignado cuando se cargó la solicitud | String
Permit_Type | Tipo del permiso, representado numericamente | Numeric
Permit_Creation_Date | Fecha en que se creó el permiso | DateTime
Block | Variable referida a la dirección | String
Lot | Variable referida a la dirección | String
Street_Number | Variable referida a la dirección | Numeric
Street_Name | Variable referida a la dirección | String
Street_Suffix | Variable referida a la dirección | String
Description | Detalles del propósito del permiso | String
Current_Status | Estado actual de la solicitud | Numeric
Current_Status_Date | Fecha de actualización del estado actual | DateTime
Filed_Date | Fecha de presentación del permiso | DateTime
Issued_Date | Fecha de emisión del permiso | DateTime
Complated_Date | Fecha de cuando el proyecto fue completado | DateTime
First_Construction_Document_Date | Fecha de cuando fue documentado la construcción | DateTime
Number_of_Existing_Stories | Número de historias existentes en el edificio | Numeric
Number_of_Proposed_Stories | Número de historias propuestas para la construcción / alteración | Numeric
Permit_Expiration_Date | Fecha de expiración del permiso | DateTime
Estimate_Cost | Estimación inicial del costo del proyecto | Numeric
Revised_Cost | Estimación revisada del costo del proyecto | Numeric
Existing_Use | Uso del edificio | String
Existing_Units | Número de unidades existentes| Numeric
Proposed_Use | Uso propuesto del edificio | String
Proposed_Units | Número de unidades propuestas | Numeric
Plansets | Representación del plan que indica el diseño general (intención de la fundación) | String
Existing_Construction_Type | Tipo de construcción existente | String
Proposed_Construction_Type | Tipo de construcción propuesto | String
Supervisor_District | Supervisor del distrito al que pertenece la ubicación del edificio | String
Neighborhoods_Analysis_Boundaries | Barrio al que pertenece la ubicación del edificio | String
Zipcode | Código postal de la dirección del edificio | Numerc
Location | Par de Latitud y Longitud de la ubicación | String


