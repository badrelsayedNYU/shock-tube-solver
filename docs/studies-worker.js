importScripts("studies-core.js", "studies-run.js");
onmessage = ({data}) => {
  try { postMessage({ done: Studies.run(data, progress => postMessage({progress})) }); }
  catch (error) { postMessage({error: error.message}); }
};
