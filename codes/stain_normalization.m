% This functions do different Stain Normalization methods
%source_img_path=The original images need to be normalized with respect to
%the reference image,TargetImage=reference image
function stain_normalization(source_img_path,TargetImage1)
%img_path_rgb=[source_img_path '_normalization_RGB histogram specification'];
%img_path_reinhard=[source_img_path '_normalization_Reinhard'];
img_path_macenko=[source_img_path '_normalization_Macenko_test'];
%mkdir(img_path_reinhard);
%mkdir(img_path_rgb);
mkdir(img_path_macenko);
verbose = 1;
TargetImage=imread(TargetImage1);
whole_image_extension='bmp';
whole_image_path=dir(fullfile([source_img_path '/' '*.' whole_image_extension]));

for whole_image_count=1:size(whole_image_path,1)
    whole_image_name_ext=whole_image_path(whole_image_count).name;
    whole_image_name=whole_image_name_ext(1:end-4);
    SourceImage=imread(fullfile([source_img_path '/' whole_image_name_ext]));
    
    %Stain Normalization using RGB Histogram specification
   %[ NormHS ]=Norm(SourceImage,TargetImage,'RGBHist',verbose );
    %imwrite(NormHS,fullfile([img_path_rgb,'/',whole_image_name,'.jpg']));

     %Stain Normalization using Reinhard method     
    %[ NormRH ] = Norm( SourceImage, TargetImage, 'Reinhard', verbose );
    %imwrite(NormRH,fullfile([img_path_reinhard,'/',whole_image_name,'.jpg']));

%% Stain Normalisation using Macenko's Method

[ NormMM ] = Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1, verbose);

 imwrite(NormMM,fullfile([img_path_macenko,'/',whole_image_name,'.bmp']));

  end
