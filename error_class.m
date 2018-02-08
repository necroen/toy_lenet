function top_err = error_class(x, labels)

predictions = gather(x{end-1}) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
top_err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;  % top1err ?
top_err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ; % top5err ?

end

